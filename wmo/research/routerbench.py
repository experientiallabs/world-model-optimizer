"""RouterBench (arXiv 2403.12031) adapter: their precomputed matrix as our `OutcomeMatrix`.

RouterBench ships 36k+ prompts x 11 models with per-call quality scores and measured dollar
costs (huggingface.co/datasets/withmartian/routerbench). Loading it into `OutcomeMatrix` lets
the routing fitter run on data with PUBLISHED baselines before we spend anything on our own
world-model matrices: if the fitter cannot land in the published cluster/kNN band here, the bug
is ours, not our data's.

Cost note: RouterBench publishes measured per-call totals, not per-Mtok prices, so the pool
snapshot carries explicit 0.0 per-Mtok prices and every outcome carries its dataset cost in
`cost_usd`. Consumers must price from OUTCOMES, never from this pool's entries (the fitter's
contract anyway).

Needs the `viz` extra (pandas); import this module lazily from CLI paths, matching
`wmo.research.concurrency_plot`.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.providers.base import ProviderKind
from wmo.providers.pool import PoolEntry

if TYPE_CHECKING:
    from pathlib import Path

# The 11 models of the 0-shot matrix, in the dataset's own column naming.
ROUTERBENCH_MODELS = [
    "WizardLM/WizardLM-13B-V1.2",
    "claude-instant-v1",
    "claude-v1",
    "claude-v2",
    "gpt-3.5-turbo-1106",
    "gpt-4-1106-preview",
    "meta/code-llama-instruct-34b-chat",
    "meta/llama-2-70b-chat",
    "mistralai/mistral-7b-chat",
    "mistralai/mixtral-8x7b-chat",
    "zero-one-ai/Yi-34B-Chat",
]


def load_routerbench(
    path: Path,
    *,
    models: list[str] | None = None,
    benchmarks: list[str] | None = None,
    sample: int | None = None,
    seed: int = 0,
    include_replies: bool = False,
) -> OutcomeMatrix:
    """Load the RouterBench pickle into an `OutcomeMatrix`.

    `benchmarks` filters by `eval_name`; `sample` draws that many prompts (deterministic in
    `seed`) AFTER filtering; `include_replies` keeps the per-model response texts (off by
    default: 36k prompts x 11 responses is most of the file's memory).
    """
    # read_pickle is typed DataFrame | Series; this dataset is always a DataFrame.
    frame = cast("pd.DataFrame", pd.read_pickle(path))
    model_names = models if models is not None else ROUTERBENCH_MODELS
    if benchmarks is not None:
        frame = frame[frame["eval_name"].isin(benchmarks)]
        if frame.empty:
            available = ", ".join(sorted(pd.read_pickle(path)["eval_name"].unique())[:20])
            raise ValueError(f"no rows for benchmarks {benchmarks}; eval names start: {available}")
    if sample is not None and sample < len(frame):
        frame = frame.sample(n=sample, random_state=seed).sort_index()

    pool = [
        PoolEntry(
            name=name,
            kind=ProviderKind.OPENAI,
            model=name,
            tier="frontier" if name == "gpt-4-1106-preview" else "open",
            # Per-Mtok prices unpublished; costs live on each outcome (see module docstring).
            input_per_mtok=0.0,
            output_per_mtok=0.0,
        )
        for name in model_names
    ]

    outcomes: list[ScenarioOutcome] = []
    # Dict records, not itertuples: the dataset's column names ("model-a", "x|total_cost")
    # are not Python identifiers, so itertuples silently positionalizes them.
    for row in frame.to_dict("records"):
        scenario_id = f"{row['eval_name']}:{row['sample_id']}"
        prompt = str(row["prompt"])
        for name in model_names:
            replies = [str(row[f"{name}|model_response"])] if include_replies else []
            reward = float(row[name])
            outcomes.append(
                ScenarioOutcome(
                    scenario_id=scenario_id,
                    task=prompt,
                    model=name,
                    reward=reward,
                    success=reward >= 0.5,
                    steps=1,
                    stop_reason="routerbench",
                    cost_usd=float(row[f"{name}|total_cost"]),
                    replies=replies,
                )
            )
    return OutcomeMatrix(pool=pool, outcomes=outcomes)


def _eval_name(scenario_id: str) -> str:
    # Ids without a "dataset:" prefix (wm matrices use raw trace ids) share ONE stratum;
    # otherwise every scenario is a size-1 stratum and the split sends them all to fit.
    return scenario_id.split(":", 1)[0] if ":" in scenario_id else ""


def split_scenario_ids(
    matrix: OutcomeMatrix, *, train_fraction: float = 0.7, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Deterministic fit/test split of scenario ids, stratified by eval name.

    Stratification mirrors the RouterBench protocol (per-dataset splits) so a small benchmark
    cannot vanish entirely from either side.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    by_eval: dict[str, list[str]] = {}
    for scenario_id in matrix.scenario_ids():
        by_eval.setdefault(_eval_name(scenario_id), []).append(scenario_id)
    rng = random.Random(seed)
    fit: list[str] = []
    test: list[str] = []
    for _name, ids in sorted(by_eval.items()):
        shuffled = ids[:]
        rng.shuffle(shuffled)
        cut = max(1, round(len(shuffled) * train_fraction)) if len(shuffled) > 1 else 1
        cut = min(cut, len(shuffled) - 1) if len(shuffled) > 1 else cut
        fit.extend(shuffled[:cut])
        test.extend(shuffled[cut:])
    return sorted(fit), sorted(test)


def _mean_by_model(matrix: OutcomeMatrix, ids: set[str]) -> dict[str, tuple[float, float]]:
    """model -> (mean reward, mean cost) over scored outcomes in `ids`."""
    sums: dict[str, tuple[float, float, int]] = {}
    for outcome in matrix.outcomes:
        if outcome.scenario_id not in ids or outcome.reward is None:
            continue
        reward_sum, cost_sum, count = sums.get(outcome.model, (0.0, 0.0, 0))
        sums[outcome.model] = (
            reward_sum + outcome.reward,
            cost_sum + outcome.cost_usd,
            count + 1,
        )
    return {
        model: (reward_sum / count, cost_sum / count)
        for model, (reward_sum, cost_sum, count) in sums.items()
        if count
    }


def best_single_model(
    matrix: OutcomeMatrix, *, fit_ids: list[str], eval_ids: list[str]
) -> tuple[str, float, float]:
    """The strongest single model in hindsight: chosen on `fit_ids`, measured on `eval_ids`.

    Ties on fit accuracy break toward the cheaper model (the honest tie: same quality for
    less). Returns (model, eval accuracy, eval mean cost).
    """
    fit_stats = _mean_by_model(matrix, set(fit_ids))
    if not fit_stats:
        raise ValueError("no scored outcomes in fit_ids; cannot choose a best single model")
    winner = min(fit_stats.items(), key=lambda kv: (-kv[1][0], kv[1][1]))[0]
    eval_stats = _mean_by_model(matrix, set(eval_ids))
    accuracy, cost = eval_stats[winner]
    return winner, accuracy, cost


def oracle(matrix: OutcomeMatrix, ids: list[str]) -> tuple[float, float]:
    """Per-scenario best model (ties toward cheaper): the routing ceiling on `ids`."""
    wanted = set(ids)
    best: dict[str, tuple[float, float]] = {}
    for outcome in matrix.outcomes:
        if outcome.scenario_id not in wanted or outcome.reward is None:
            continue
        incumbent = best.get(outcome.scenario_id)
        candidate = (outcome.reward, outcome.cost_usd)
        if (
            incumbent is None
            or candidate[0] > incumbent[0]
            or (candidate[0] == incumbent[0] and candidate[1] < incumbent[1])
        ):
            best[outcome.scenario_id] = candidate
    if not best:
        raise ValueError("no scored outcomes in ids; cannot compute the oracle")
    rewards = [reward for reward, _cost in best.values()]
    costs = [cost for _reward, cost in best.values()]
    return sum(rewards) / len(rewards), sum(costs) / len(costs)


def random_baseline(matrix: OutcomeMatrix, ids: list[str]) -> tuple[float, float]:
    """Expected accuracy/cost of uniform-random assignment over the pool (closed form)."""
    stats = _mean_by_model(matrix, set(ids))
    if not stats:
        raise ValueError("no scored outcomes in ids; cannot compute the random baseline")
    rewards = [reward for reward, _cost in stats.values()]
    costs = [cost for _reward, cost in stats.values()]
    return sum(rewards) / len(rewards), sum(costs) / len(costs)


def upper_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Non-decreasing upper concave envelope of (cost, quality) points.

    The curve AIQ integrates (RouterBench evaluation/AIQ.py): sort by cost, keep the upper
    convex chain, and drop any point whose quality dips below its predecessor (a rational
    operator never pays more for less).
    """
    if not points:
        raise ValueError("no points to hull")
    ordered = sorted(points)
    chain: list[tuple[float, float]] = []
    for point in ordered:
        while len(chain) >= 2:
            (x1, y1), (x2, y2) = chain[-2], chain[-1]
            # Pop the middle point when it sits on or under the chord to the new point.
            if (y2 - y1) * (point[0] - x1) <= (point[1] - y1) * (x2 - x1):
                chain.pop()
            else:
                break
        chain.append(point)
    monotone: list[tuple[float, float]] = []
    for point in chain:
        if monotone and point[1] < monotone[-1][1]:
            continue
        monotone.append(point)
    return monotone


def aiq(points: list[tuple[float, float]], *, max_cost: float) -> float:
    """Area under the hull, extended flat to `max_cost`, normalized by `max_cost`.

    Matches RouterBench's calc_AIQ. NOTE: the normalization couples AIQ to the comparison
    set's most expensive point, so absolute AIQs are only comparable when computed against
    the same `max_cost`; always report the hull points alongside.
    """
    hull = upper_hull(points)
    if hull[-1][0] < max_cost:
        hull = [*hull, (max_cost, hull[-1][1])]
    xs = [x for x, _y in hull]
    ys = [y for _x, y in hull]
    area = float(np.trapezoid(ys, xs))
    return area / max_cost


def single_model_points(matrix: OutcomeMatrix, ids: list[str]) -> dict[str, tuple[float, float]]:
    """Each model's (mean cost, mean reward) on `ids`: the Zero-Router's constituent points.

    RouterBench's non-predictive floor is the hull of these (random mixing between two models
    traces the segment joining them, so the hull IS the best label-free strategy).
    """
    stats = _mean_by_model(matrix, set(ids))
    if not stats:
        raise ValueError("no scored outcomes in ids")
    return {model: (cost, reward) for model, (reward, cost) in sorted(stats.items())}
