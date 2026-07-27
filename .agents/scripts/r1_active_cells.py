"""Drawing-board-2 round 1: active cell selection replayed on the real tau-bench matrix.

Question: when real-harness trials cost money, which (model, scenario, episode) cells should
a capture round buy? Replay on the COMPLETE tau-bench-real matrix (9 models x 40 scenarios
x 2 episodes, all scored) as ground truth: a simulated capture strategy "buys" cells one
episode at a time (sampling the cell's real episodes without replacement, bootstrap with
replacement beyond 2) and at each budget we score what the purchased evidence supports:

- IDENTIFY: does argmax of measured means equal the true best single model (fable-5)?
- REGRET: true accuracy gap between the identified model and the true best.
- PIN-SAFETY: probability the identified pin is within 2pt of the true best.

Strategies (all start with one warm episode per model on a shared scenario batch):
- uniform: random unmeasured cells.
- grid: scenario-major round robin (the naive capture order used today).
- sh: sequential halving over MODELS (Karnin et al. ICML'13): split the remaining budget
  into log2(9) rounds; each round spends uniformly over surviving models, then drops the
  bottom half by measured mean.
- ucbe: variance-adaptive allocation (UCB-E-flavored): pull the model with the highest
  mean + sqrt(a / n) exploration bonus, scenarios chosen least-measured-first.

Costs are proportional to episodes (the real capture cost ~$159 for 720 episodes), so
budget fraction = dollar fraction. 400 replay seeds per strategy x budget.

$0, offline. Results -> findings/r1.md (drawing-board 2); no runs.jsonl rows (this is a
measurement-protocol experiment, not a routing variant).

Usage: uv run python .agents/scripts/r1_active_cells.py
"""

from __future__ import annotations

import logging
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from wmo.optimize.outcomes import OutcomeMatrix
from wmo.research.routing_corpus import routing_data

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("r1.cells")

DATA = routing_data()
REPLAYS = 400
BUDGETS = [45, 72, 108, 144, 216, 288, 360, 540]  # episodes; full matrix = 720


def load_truth(
    stem: str = "tau-bench-real",
) -> tuple[list[str], list[str], dict[tuple[int, int], list[float]]]:
    matrix = OutcomeMatrix.load(DATA / "matrices" / f"{stem}_matrix.json")
    models = [entry.name for entry in matrix.pool]
    scenarios = matrix.scenario_ids()
    cells: dict[tuple[int, int], list[float]] = defaultdict(list)
    for outcome in matrix.outcomes:
        if outcome.reward is None:
            continue
        cells[(scenarios.index(outcome.scenario_id), models.index(outcome.model))].append(
            outcome.reward
        )
    return models, scenarios, dict(cells)


class Replay:
    """One simulated capture: buys episodes, tracks measured means per model."""

    def __init__(self, cells: dict[tuple[int, int], list[float]], n_models: int, seed: int):
        self.rng = random.Random(seed)
        # Per-replay shuffle of each cell's episode order: a real capture draws from the
        # episode distribution, it does not replay the stored file order (without this, a
        # strategy that covers every cell exactly once is deterministic across replays).
        self.cells = {key: self.rng.sample(values, len(values)) for key, values in cells.items()}
        self.n_models = n_models
        self.drawn: dict[tuple[int, int], int] = defaultdict(int)
        self.sums = np.zeros(n_models)
        self.counts = np.zeros(n_models)

    def buy(self, scenario: int, model: int) -> None:
        episodes = self.cells.get((scenario, model))
        if not episodes:
            return
        taken = self.drawn[(scenario, model)]
        value = (
            episodes[taken] if taken < len(episodes) else self.rng.choice(episodes)
        )  # bootstrap past the 2 real episodes
        self.drawn[(scenario, model)] += 1
        self.sums[model] += value
        self.counts[model] += 1

    def means(self) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            return np.where(self.counts > 0, self.sums / np.maximum(self.counts, 1), -1.0)


def run_strategy(
    name: str,
    cells: dict[tuple[int, int], list[float]],
    n_models: int,
    n_scenarios: int,
    budget: int,
    seed: int,
) -> int:
    replay = Replay(cells, n_models, seed)
    rng = replay.rng
    scenario_order = list(range(n_scenarios))
    rng.shuffle(scenario_order)

    if name == "uniform":
        pairs = [(s, m) for s in range(n_scenarios) for m in range(n_models)]
        rng.shuffle(pairs)
        for scenario, model in (pairs * 2)[:budget]:
            replay.buy(scenario, model)
    elif name == "grid":
        spent = 0
        episode = 0
        while spent < budget:
            for scenario in scenario_order:
                for model in range(n_models):
                    if spent >= budget:
                        break
                    replay.buy(scenario, model)
                    spent += 1
                if spent >= budget:
                    break
            episode += 1
    elif name == "sh":
        alive = list(range(n_models))
        rounds = max(1, math.ceil(math.log2(n_models)))
        per_round = budget // rounds
        cursor = 0
        for _ in range(rounds):
            if len(alive) == 1:
                break
            pulls = max(1, per_round // len(alive))
            for model in alive:
                for _ in range(pulls):
                    replay.buy(scenario_order[cursor % n_scenarios], model)
                    cursor += 1
            means = replay.means()
            alive = sorted(alive, key=lambda m: -means[m])[: max(1, len(alive) // 2)]
    elif name == "ucbe":
        # warm start: one episode per model
        for index, model in enumerate(range(n_models)):
            replay.buy(scenario_order[index % n_scenarios], model)
        a = 2.0  # exploration scale, budget-order per Audibert & Bubeck's a ~ B/K guidance
        for step in range(budget - n_models):
            means = replay.means()
            bonus = np.sqrt((a * budget / n_models) / np.maximum(replay.counts, 1))
            model = int(np.argmax(means + bonus))
            replay.buy(scenario_order[step % n_scenarios], model)
    elif name == "pilot":
        # Pilot-then-commit (round-1 protocol recommendation, now measured): spend ~10% as
        # a uniform grid, test the top-2 gap against its SE, then commit the remainder to
        # sequential halving when the gap is clear and to the grid when it is not. Pilot
        # evidence carries into the committed phase (same replay object).
        pilot = min(budget, max(n_models * 4, int(0.1 * 720)))
        spent = 0
        while spent < pilot:
            for scenario in scenario_order:
                for model in range(n_models):
                    if spent >= pilot:
                        break
                    replay.buy(scenario, model)
                    spent += 1
                if spent >= pilot:
                    break
        means = replay.means()
        order = np.argsort(means)
        top, second = int(order[-1]), int(order[-2])
        per_model = max(1, pilot // n_models)
        # Binary-ish rewards: bound each mean's variance by p(1-p)/n, gap SE by the sum.
        gap = float(means[top] - means[second])
        se_gap = float(
            np.sqrt(sum(max(m * (1 - m), 0.05) / per_model for m in (means[top], means[second])))
        )
        remainder = budget - spent
        if gap > 2.0 * se_gap:
            alive = list(range(n_models))
            rounds = max(1, math.ceil(math.log2(n_models)))
            per_round = remainder // rounds if remainder else 0
            cursor = 0
            for _ in range(rounds):
                if len(alive) == 1 or per_round == 0:
                    break
                pulls = max(1, per_round // len(alive))
                for model in alive:
                    for _ in range(pulls):
                        replay.buy(scenario_order[cursor % n_scenarios], model)
                        cursor += 1
                current = replay.means()
                alive = sorted(alive, key=lambda m: -current[m])[: max(1, len(alive) // 2)]
        else:
            while spent < budget:
                for scenario in scenario_order:
                    for model in range(n_models):
                        if spent >= budget:
                            break
                        replay.buy(scenario, model)
                        spent += 1
                    if spent >= budget:
                        break
    else:
        raise ValueError(name)
    return int(np.argmax(replay.means()))


def main() -> None:
    import sys

    stem = next((a for a in sys.argv[1:] if not a.startswith("-")), "tau-bench-real")
    models, scenarios, cells = load_truth(stem)
    logger.info("== %s ==", stem)
    true_means = np.zeros(len(models))
    for model in range(len(models)):
        values = [v for (s, m), vs in cells.items() if m == model for v in vs]
        true_means[model] = float(np.mean(values))
    best = int(np.argmax(true_means))
    logger.info(
        "truth: best=%s %.4f | runner-up gap %.4f | full matrix = 720 episodes",
        models[best],
        true_means[best],
        true_means[best] - float(np.sort(true_means)[-2]),
    )

    strategies = ("uniform", "grid", "sh", "ucbe", "pilot")
    header = f"{'budget':>7s} ({'%full':>5s})"
    for name in strategies:
        header += f" | {name}: P(best) regret safe2pt"
    logger.info(header)
    for budget in BUDGETS:
        line = f"{budget:7d} ({budget / 720:5.0%})"
        for name in strategies:
            hits = 0
            safe = 0
            regrets = []
            for seed in range(REPLAYS):
                picked = run_strategy(name, cells, len(models), len(scenarios), budget, seed)
                hits += int(picked == best)
                regret = true_means[best] - true_means[picked]
                safe += int(regret <= 0.02)
                regrets.append(regret)
            line += f" | {hits / REPLAYS:4.2f} {np.mean(regrets):+.4f} {safe / REPLAYS:4.2f}"
        logger.info(line)


if __name__ == "__main__":
    main()
