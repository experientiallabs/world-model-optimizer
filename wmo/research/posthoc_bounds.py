"""Post-hoc routing bounds: what a verifier that reads the ANSWER could buy, and what free
features actually deliver.

Every router family we have fitted so far (rank/cluster, kNN profile, IRT ability) predicts a
model from the QUERY. This module measures the other axis on the same precomputed matrices: given
that a cell already holds several episodes, how much reward is recoverable by SELECTING among
completed rollouts rather than by picking the model up front?

Three quantities, in the order they should be read:

1. `corpus_bounds` - the best-of-n ceiling. `oracle_of_n` per model takes the max reward over that
   cell's episodes at the sum of their costs. It is an ORACLE (a perfect verifier), so it bounds
   best-of-n from above; on several corpora a cheap model's best-of-2 clears the best single model
   on BOTH accuracy and cost, which is the reason the family is worth measuring at all.
2. `selector_bounds` - what FREE post-hoc features (did it finish, how many steps, how many output
   tokens) recover of that ceiling. Measured as within-cell selection: same scenario, same model,
   pick one of its episodes. This is the number that decides whether best-of-n is cheap.
3. `feature_correlations` - why (1) and (2) disagree. Each feature's pooled correlation with reward
   is decomposed into a BETWEEN-cell part (scenario/model difficulty: hard scenarios truncate and
   score low) and a WITHIN-cell part (rollout quality at fixed difficulty, the only part a
   selector can use). A strong pooled correlation that is entirely between-cell is not a verifier
   signal, and reading the pooled number alone will mislead.

CAVEAT that governs every number here: `oracle_of_n` capitalizes on episode-to-episode reward
variance, and this harness cannot tell rollout variance from JUDGE variance. Where episodes
disagree because the judge scored the same quality of work differently, the ceiling is phantom and
no verifier can reach it. `episode_disagreement_*` reports the raw spread; separating its two
components needs repeated judging of one stored reply, which is a different experiment.
"""

from __future__ import annotations

import statistics as st
from typing import TYPE_CHECKING

from pydantic import BaseModel

from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

# Free post-hoc scalars: available the moment a rollout finishes, at zero cost and zero latency.
# Signed so that LOWER is the "looks better" direction, which is what the selectors below assume.
SCALAR_FEATURES: dict[str, Callable[[ScenarioOutcome], float]] = {
    "steps": lambda o: float(o.steps),
    "stop_is_max_steps": lambda o: 1.0 if o.stop_reason == "max_steps" else 0.0,
    "reply_chars": lambda o: float(sum(len(reply) for reply in o.replies)),
    "output_tokens": lambda o: float(o.usage.output_tokens),
    "n_replies": lambda o: float(len(o.replies)),
}

# Within-cell selection keys, compared lexicographically; ties are treated as "cannot distinguish".
#
# READ THE SIGNS. The pooled correlation between effort (steps, replies, tokens) and reward is
# NEGATIVE, but that is between-cell difficulty: hard scenarios burn steps and score low. WITHIN a
# cell the sign FLIPS (tau-bench steps +0.310, n_replies +0.363), because at fixed difficulty the
# rollout that did more work is the better one. A selector built on the pooled sign is therefore
# anti-correlated and loses to a coin flip; the `fewer-*` keys below are kept only to demonstrate
# that failure. `finished-then-more-steps` is the one that works.
SELECTOR_KEYS: dict[str, Callable[[ScenarioOutcome], tuple[float, ...]]] = {
    # Two independently verified components: prefer an episode that terminated normally over one
    # that hit the step cap, then prefer the one that did more work. See DEFAULT_SELECTOR.
    "finished-then-more-steps": lambda o: (
        1.0 if o.stop_reason == "max_steps" else 0.0,
        -float(o.steps),
    ),
    "more-replies": lambda o: (-float(len(o.replies)),),
    "more-output-tokens": lambda o: (-float(o.usage.output_tokens),),
    # Pooled-sign selectors: these LOSE to chance. Kept as the negative control.
    "fewer-steps (pooled-sign control)": lambda o: (float(o.steps),),
    "fewer-output-tokens (pooled-sign control)": lambda o: (float(o.usage.output_tokens),),
}

# The free selector the achievable best-of-n numbers are reported at. Both of its components are
# separately significant on the wm corpora (pooled over corpora, vs a coin flip): the finish term
# at 69.7% correct on 165 decisive cells (z=+5.06), and the effort term at 66.2% on 275 cells where
# BOTH episodes finished (z=+5.37), the latter being the control that rules out the step cap as the
# whole explanation. The finish term will weaken as max_steps rises; the effort term is
# cap-independent. All three z values come from `pooled_correct_z`, i.e. the null standard error.
DEFAULT_SELECTOR = "finished-then-more-steps"


class BestOfNBound(BaseModel):
    """Per-model best-of-n ceiling and achievable point against the corpus's best single model.

    `oracle_of_n_*` assumes a perfect verifier and is the upper bound. `selected_of_n_accuracy` is
    what `DEFAULT_SELECTOR` actually reaches using free post-hoc features - no extra call, no added
    latency - and is the number to quote as reachable today. Both share `oracle_of_n_cost_per_call`,
    since running n episodes costs the same however you choose among them.
    """

    model: str
    cells: int  # scenarios with a full complement of episodes
    episodes: int  # episodes per cell this bound was computed at (the n in best-of-n)
    one_shot_accuracy: float
    one_shot_cost_per_call: float
    oracle_of_n_accuracy: float
    oracle_of_n_cost_per_call: float
    selected_of_n_accuracy: float
    beats_best_single_accuracy: bool  # on the ACHIEVABLE point, not the oracle
    beats_best_single_cost: bool


class CorpusBounds(BaseModel):
    """Best-single / cross-model-oracle anchors plus the per-model best-of-n ceiling."""

    corpus: str
    scenarios: int
    models: int
    episodes_per_cell: list[int]  # sorted distinct episode counts; [1] means best-of-n is undefined
    best_single: str
    best_single_accuracy: float
    best_single_cost_per_call: float
    oracle_accuracy: float  # cross-model: per scenario, the best model's mean reward
    # Raw episode spread per cell (max - min). `None` when no cell has >1 episode.
    episode_disagreement_mean: float | None = None
    episode_disagreement_fraction: float | None = None
    best_of_n: list[BestOfNBound] = []


class SelectorBound(BaseModel):
    """What one free feature recovers of the within-cell best-of-n ceiling."""

    corpus: str
    feature: str
    cells: int  # cells with >1 episode that passed the filter
    random_of_n: float  # expected reward of taking an arbitrary episode
    selector_accuracy: float
    oracle_of_n: float
    decisive_cells: int  # cells where the feature ranks episodes AND rewards differ
    correct_fraction: float  # of decisive cells, how often the pick was a best episode
    harvested_fraction: float  # (selector - random) / (oracle - random); 0.0 == chance


class FeatureCorrelation(BaseModel):
    """Pooled correlation of a feature with reward, split into between- and within-cell parts."""

    corpus: str
    feature: str
    pooled: float
    between_cell: float  # scenario/model difficulty - NOT usable by a selector
    within_cell: float  # rollout quality at fixed difficulty - the verifier-relevant part


def _correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson r, or 0.0 when either side is constant (no signal rather than a division error)."""
    if len(xs) < 2:
        return 0.0
    mean_x, mean_y = st.mean(xs), st.mean(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    dev_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    dev_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5
    if dev_x == 0.0 or dev_y == 0.0:
        return 0.0
    return cov / (dev_x * dev_y)


def scored_cells(matrix: OutcomeMatrix) -> dict[tuple[str, str], list[ScenarioOutcome]]:
    """Group scored episodes by (scenario_id, model). Unscored rows are dropped, never zeroed."""
    cells: dict[tuple[str, str], list[ScenarioOutcome]] = {}
    for outcome in matrix.outcomes:
        if outcome.reward is not None:
            cells.setdefault((outcome.scenario_id, outcome.model), []).append(outcome)
    return cells


def _cell_reward(episodes: Iterable[ScenarioOutcome]) -> float:
    rewards = [o.reward for o in episodes if o.reward is not None]
    return sum(rewards) / len(rewards)


def _cell_cost(episodes: Iterable[ScenarioOutcome]) -> float:
    costs = [o.cost_usd for o in episodes]
    return sum(costs) / len(costs)


def _selected_reward(
    episodes: Sequence[ScenarioOutcome], key: Callable[[ScenarioOutcome], tuple[float, ...]]
) -> float:
    """Reward of the episode `key` ranks first; the mean over tied episodes when it cannot rank."""
    best = min(key(o) for o in episodes)
    picked = [o.reward for o in episodes if key(o) == best and o.reward is not None]
    return sum(picked) / len(picked)


def best_of_n_by_model(
    matrix: OutcomeMatrix,
    ids: Sequence[str] | None = None,
    *,
    baseline_accuracy: float,
    baseline_cost: float,
    selector: str = DEFAULT_SELECTOR,
    key: Callable[[ScenarioOutcome], tuple[float, ...]] | None = None,
    depth: int | None = None,
) -> list[BestOfNBound]:
    """Per-model best-of-n ceiling and achievable point over `ids` (all scenarios when None).

    This is the primitive behind both `corpus_bounds` (which passes the whole matrix) and the
    fit-split model discovery a deployable best-of-n router needs (which passes only fit ids, so
    the chosen model is never selected on test data). Returns [] when the subset has no cell
    sampled more than once, i.e. best-of-n is undefined there.

    `depth` defaults to the modal episode count over the subset: cells carrying fewer episodes are
    skipped for that model (reported in `cells`) rather than padded, so the ceiling is never
    inflated by a model that happened to be sampled more often. `beats_best_single_accuracy`
    compares the ACHIEVABLE point against `baseline_accuracy`, not the oracle.

    `key` overrides the named `selector` with any ranking callable, which is how a fitted verifier
    (`wmo.research.reply_verifier`) is scored through this same path as the free features.
    """
    key = key if key is not None else SELECTOR_KEYS[selector]
    cells = scored_cells(matrix)
    wanted = set(ids) if ids is not None else {sid for sid, _ in cells}
    cells = {(sid, model): episodes for (sid, model), episodes in cells.items() if sid in wanted}
    if not cells:
        return []
    scenario_ids = sorted({sid for sid, _ in cells})
    models = sorted({model for _, model in cells})
    if depth is None:
        depth = st.mode(len(episodes) for episodes in cells.values())
    if depth < 2:
        return []

    bounds: list[BestOfNBound] = []
    for model in models:
        present = [cells[(sid, model)] for sid in scenario_ids if (sid, model) in cells]
        full = [episodes for episodes in present if len(episodes) >= depth]
        if not full:
            continue
        oracle_rewards = [
            max(o.reward for o in episodes[:depth] if o.reward is not None) for episodes in full
        ]
        oracle_costs = [sum(o.cost_usd for o in episodes[:depth]) for episodes in full]
        selected_rewards = [_selected_reward(episodes[:depth], key) for episodes in full]
        oracle_cost = sum(oracle_costs) / len(oracle_costs)
        selected_accuracy = sum(selected_rewards) / len(selected_rewards)
        bounds.append(
            BestOfNBound(
                model=model,
                cells=len(full),
                episodes=depth,
                one_shot_accuracy=sum(_cell_reward(e) for e in present) / len(present),
                one_shot_cost_per_call=sum(_cell_cost(e) for e in present) / len(present),
                oracle_of_n_accuracy=sum(oracle_rewards) / len(oracle_rewards),
                oracle_of_n_cost_per_call=oracle_cost,
                selected_of_n_accuracy=selected_accuracy,
                beats_best_single_accuracy=selected_accuracy > baseline_accuracy,
                beats_best_single_cost=oracle_cost < baseline_cost,
            )
        )
    bounds.sort(key=lambda bound: -bound.selected_of_n_accuracy)
    return bounds


def corpus_bounds(
    matrix: OutcomeMatrix,
    corpus: str,
    selector: str = DEFAULT_SELECTOR,
) -> CorpusBounds:
    """Anchors + per-model best-of-n ceiling and achievable point for one whole matrix."""
    cells = scored_cells(matrix)
    if not cells:
        raise ValueError(f"matrix '{corpus}' has no scored outcomes")
    scenario_ids = sorted({sid for sid, _ in cells})
    models = sorted({model for _, model in cells})
    counts = sorted({len(episodes) for episodes in cells.values()})

    def mean_reward(model: str) -> float:
        rewards = [
            _cell_reward(cells[(sid, model)]) for sid in scenario_ids if (sid, model) in cells
        ]
        return sum(rewards) / len(rewards) if rewards else 0.0

    def mean_cost(model: str) -> float:
        costs = [_cell_cost(cells[(sid, model)]) for sid in scenario_ids if (sid, model) in cells]
        return sum(costs) / len(costs) if costs else 0.0

    best_single = max(models, key=mean_reward)
    best_accuracy = mean_reward(best_single)
    best_cost = mean_cost(best_single)

    per_scenario_best = [
        max(_cell_reward(cells[(sid, model)]) for model in models if (sid, model) in cells)
        for sid in scenario_ids
    ]
    oracle = sum(per_scenario_best) / len(per_scenario_best)

    spreads = [
        max(o.reward for o in episodes if o.reward is not None)
        - min(o.reward for o in episodes if o.reward is not None)
        for episodes in cells.values()
        if len(episodes) > 1
    ]

    bounds = best_of_n_by_model(
        matrix,
        baseline_accuracy=best_accuracy,
        baseline_cost=best_cost,
        selector=selector,
    )

    return CorpusBounds(
        corpus=corpus,
        scenarios=len(scenario_ids),
        models=len(models),
        episodes_per_cell=counts,
        best_single=best_single,
        best_single_accuracy=best_accuracy,
        best_single_cost_per_call=best_cost,
        oracle_accuracy=oracle,
        episode_disagreement_mean=st.mean(spreads) if spreads else None,
        episode_disagreement_fraction=(
            sum(1 for spread in spreads if spread > 0.0) / len(spreads) if spreads else None
        ),
        best_of_n=bounds,
    )


def selector_bound(
    matrix: OutcomeMatrix,
    corpus: str,
    feature: str,
    key: Callable[[ScenarioOutcome], tuple[float, ...]],
    *,
    cell_filter: Callable[[Sequence[ScenarioOutcome]], bool] | None = None,
) -> SelectorBound:
    """Score one within-cell selector: same scenario, same model, pick among its episodes.

    Ties on `key` mean the feature cannot distinguish the episodes, so the cell contributes its
    mean reward (the expected value of an arbitrary pick) and is not counted as decisive. A cell is
    decisive only when the feature ranks the episodes AND their rewards differ - the only cells
    where the selector can be right or wrong.

    `cell_filter` restricts which cells count, which is how the confound controls are expressed:
    filtering to cells where every episode terminated normally holds the step cap constant and
    isolates the effort signal from the finish signal.
    """
    cells = scored_cells(matrix)
    random_rewards: list[float] = []
    selected_rewards: list[float] = []
    oracle_rewards: list[float] = []
    decisive = 0
    correct = 0
    for episodes in cells.values():
        if len(episodes) < 2 or (cell_filter is not None and not cell_filter(episodes)):
            continue
        rewards = [o.reward for o in episodes if o.reward is not None]
        mean = sum(rewards) / len(rewards)
        random_rewards.append(mean)
        oracle_rewards.append(max(rewards))
        best_key = min(key(o) for o in episodes)
        picked = [o.reward for o in episodes if key(o) == best_key and o.reward is not None]
        selected_rewards.append(sum(picked) / len(picked))
        ranks_episodes = len({key(o) for o in episodes}) > 1
        rewards_differ = max(rewards) > min(rewards)
        if ranks_episodes and rewards_differ:
            decisive += 1
            if max(picked) == max(rewards):
                correct += 1
    if not random_rewards:
        raise ValueError(f"matrix '{corpus}' has no cell with more than one scored episode")

    random_of_n = st.mean(random_rewards)
    selector = st.mean(selected_rewards)
    oracle = st.mean(oracle_rewards)
    headroom = oracle - random_of_n
    return SelectorBound(
        corpus=corpus,
        feature=feature,
        cells=len(random_rewards),
        random_of_n=random_of_n,
        selector_accuracy=selector,
        oracle_of_n=oracle,
        decisive_cells=decisive,
        correct_fraction=correct / decisive if decisive else 0.0,
        harvested_fraction=(selector - random_of_n) / headroom if headroom > 0.0 else 0.0,
    )


def all_finished(episodes: Sequence[ScenarioOutcome]) -> bool:
    """Cell filter: every episode terminated normally, so the step cap is constant within it."""
    return all(o.stop_reason != "max_steps" for o in episodes)


def one_finished(episodes: Sequence[ScenarioOutcome]) -> bool:
    """Cell filter: exactly one episode terminated normally - the pure step-cap contrast."""
    return sum(1 for o in episodes if o.stop_reason != "max_steps") == 1


def pooled_correct_z(bounds: Iterable[SelectorBound]) -> tuple[int, int, float]:
    """Pool decisive picks across corpora: (correct, decisive, z against a coin flip).

    Per-corpus decisive counts here are 10-150, too small to read individually; the pooled z is
    what says whether a selector beats chance. Returns z=0.0 when nothing was decisive.

    The standard error is the NULL one, sqrt(0.25 / n), not the observed-proportion
    sqrt(p(1-p) / n): the test is against a fixed p=0.5, so the null SD is the correct
    denominator. The observed-proportion form runs ~5-10% larger here and inflates z.
    """
    decisive = 0
    correct = 0
    for bound in bounds:
        decisive += bound.decisive_cells
        correct += round(bound.correct_fraction * bound.decisive_cells)
    if decisive == 0:
        return 0, 0, 0.0
    rate = correct / decisive
    standard_error = (0.25 / decisive) ** 0.5  # under the null, p = 0.5
    return correct, decisive, (rate - 0.5) / standard_error


def feature_correlations(matrix: OutcomeMatrix, corpus: str) -> list[FeatureCorrelation]:
    """Decompose each free feature's correlation with reward into between- and within-cell parts.

    The between-cell part correlates per-cell means (one point per scenario x model) and measures
    difficulty. The within-cell part correlates deviations from each cell's own means and measures
    rollout quality at fixed difficulty. Only the latter is available to a selector.
    """
    cells = scored_cells(matrix)
    results: list[FeatureCorrelation] = []
    for name, extract in SCALAR_FEATURES.items():
        pooled_x: list[float] = []
        pooled_y: list[float] = []
        between_x: list[float] = []
        between_y: list[float] = []
        within_x: list[float] = []
        within_y: list[float] = []
        for episodes in cells.values():
            values = [extract(o) for o in episodes]
            rewards = [o.reward for o in episodes if o.reward is not None]
            pooled_x.extend(values)
            pooled_y.extend(rewards)
            between_x.append(st.mean(values))
            between_y.append(st.mean(rewards))
            if len(episodes) > 1:
                mean_value, mean_reward = st.mean(values), st.mean(rewards)
                within_x.extend(value - mean_value for value in values)
                within_y.extend(reward - mean_reward for reward in rewards)
        results.append(
            FeatureCorrelation(
                corpus=corpus,
                feature=name,
                pooled=_correlation(pooled_x, pooled_y),
                between_cell=_correlation(between_x, between_y),
                within_cell=_correlation(within_x, within_y),
            )
        )
    return results
