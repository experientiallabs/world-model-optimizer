"""Paired per-scenario statistics for the corner analyses: the binding conventions in code form.

Owned by the quality corner chat (the charter assigns it the statistical rigor for all three
chats); the latency and cost chats consume these helpers and extend them here, never fork them.

Three conventions, enforced by shape rather than by memory:

1. PAIRED, PER SCENARIO. `paired_delta` only accepts two per-scenario reward maps and works on
   the intersection of scenarios scored on both sides. There is no unpaired two-sample helper
   in this module on purpose: scenario difficulty dominates lever effects at this program's
   sample sizes, and pairing is what removes it.
2. THE NOISE FLOOR. Paired mean-reward deltas within +-0.015 to 0.02 at these sample sizes are
   indistinguishable from noise (measured by the model-refresh round, carried forward by the
   tau grid forecast). `PairedDelta.within_noise_floor` travels with every delta so a renderer
   cannot forget to say so.
3. SIM-TO-REAL FRAMING. Per-scenario paired sign agreement (`sign_agreement`) is the primary
   transfer evidence. A model-mean rank correlation over a handful of models sits inside its
   own null band (SD ~0.4-0.5 at n=4-11), so `spearman_model_means` returns a result carrying
   a descriptive-only caveat as data; quote it with the caveat or not at all.

Pure stdlib plus pydantic: no numpy and no viz extra, so every corner chat can run it anywhere
the repo installs.
"""

from __future__ import annotations

from math import comb, fsum
from random import Random
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# The conservative noise floor for paired mean-reward deltas at this program's sample sizes
# (20 scenarios x 2 episodes on the grid; 20 holdout tasks x 3 attempts on cycle rows). Deltas
# inside the band are noise; findings must not headline them. 0.015 is the optimistic end of
# the same measured range and exists for sensitivity notes, not for claims.
NOISE_FLOOR_REWARD = 0.02
NOISE_FLOOR_REWARD_OPTIMISTIC = 0.015

# Bootstrap defaults: percentile CIs, resampling SCENARIOS (the unit difficulty varies over),
# seeded so every rerun of a figure script reproduces its numbers exactly.
BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_CONFIDENCE = 0.95

# Per-scenario deltas closer to zero than this are ties: rewards are means of judge scores in
# [0, 1], so a genuine tie is an exact float match and the epsilon only absorbs summation order.
TIE_EPSILON = 1e-12


class MeanCI(BaseModel):
    """A per-scenario mean with its cluster-bootstrap CI and the sample sizes behind it."""

    mean: float
    ci_low: float
    ci_high: float
    n_scenarios: int
    n_episodes: int
    confidence: float


class PairedDelta(BaseModel):
    """A paired per-scenario delta between two arms, with everything a claim about it needs.

    `scenario_deltas` keeps the per-scenario values so downstream sign-agreement checks reuse
    exactly the pairs this summary was computed from instead of recomputing their own.
    """

    mean_delta: float
    ci_low: float
    ci_high: float
    n_pairs: int
    n_up: int
    n_down: int
    n_tied: int
    # None when no scenario moved at all: a sign test over zero movers is not evidence of
    # anything and quoting p=1.0 would dress that up as a computed result.
    sign_test_p: float | None
    within_noise_floor: bool
    noise_floor: float
    # The one-word honest reading, shared so three chats cannot phrase the same delta three
    # ways: "measurable" requires BOTH a CI that excludes zero AND a mean outside the noise
    # floor; a CI spanning zero is "no_effect" no matter how large the mean (cycle-1's -0.067
    # with CI [-0.167, +0.033] is the canonical example: noise, not a regression).
    verdict: Literal["no_effect", "within_noise_floor", "measurable"]
    scenario_deltas: dict[str, float]


class SignAgreement(BaseModel):
    """Per-scenario paired sign agreement between two legs (the primary sim-to-real evidence)."""

    agree: int
    compared: int
    ties_excluded: int
    # None when every shared scenario was a tie on one side or the other.
    fraction: float | None


class DescriptiveSpearman(BaseModel):
    """A model-mean rank correlation that carries its own descriptive-only caveat.

    The caveat is data, not documentation: any surface that quotes `rho` has the sentence in
    hand and no excuse to drop it.
    """

    rho: float
    n: int
    caveat: str = Field(min_length=1)


def mean_with_ci(
    rewards_by_scenario: Mapping[str, Sequence[float]],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
    confidence: float = DEFAULT_CONFIDENCE,
) -> MeanCI:
    """Mean reward averaged per scenario then across scenarios, with a cluster-bootstrap CI.

    Episodes within a scenario are correlated (same task, same environment), so the bootstrap
    resamples SCENARIOS, not episodes: each resample draws scenarios with replacement and takes
    the mean of their per-scenario means. This matches the scorecard's per-scenario averaging
    (wmo.optimize.scorecard) so a CI here brackets exactly the number a scorecard reports.

    Raises:
        ValueError: on an empty map or a scenario with no rewards; feed only scored episodes
            (an unscored episode is an infrastructure failure, not a 0).
    """
    means = _scenario_means(rewards_by_scenario)
    values = list(means.values())
    low, high = _bootstrap_ci(values, resamples=resamples, seed=seed, confidence=confidence)
    return MeanCI(
        mean=fsum(values) / len(values),
        ci_low=low,
        ci_high=high,
        n_scenarios=len(values),
        n_episodes=sum(len(v) for v in rewards_by_scenario.values()),
        confidence=confidence,
    )


def paired_delta(
    arm: Mapping[str, Sequence[float]],
    anchor: Mapping[str, Sequence[float]],
    *,
    noise_floor: float = NOISE_FLOOR_REWARD,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
    confidence: float = DEFAULT_CONFIDENCE,
) -> PairedDelta:
    """Paired per-scenario delta (arm minus anchor) over the scenarios scored on BOTH sides.

    Per-scenario means first, then the pairing, then a bootstrap over the paired deltas and an
    exact two-sided sign test over the scenarios that moved. Positive means the arm is better.

    Raises:
        ValueError: when the two sides share no scenario; that is two experiments, not a pair.
    """
    arm_means = _scenario_means(arm)
    anchor_means = _scenario_means(anchor)
    shared = sorted(set(arm_means) & set(anchor_means))
    if not shared:
        raise ValueError(
            f"no scenario was scored on both sides (arm has {len(arm_means)}, anchor has "
            f"{len(anchor_means)}); a paired delta needs shared scenarios, so check that both "
            f"maps key on the same scenario ids"
        )
    deltas = {sid: arm_means[sid] - anchor_means[sid] for sid in shared}
    values = list(deltas.values())
    low, high = _bootstrap_ci(values, resamples=resamples, seed=seed, confidence=confidence)
    n_up = sum(1 for d in values if d > TIE_EPSILON)
    n_down = sum(1 for d in values if d < -TIE_EPSILON)
    mean_delta = fsum(values) / len(values)
    if low <= 0.0 <= high:
        verdict: Literal["no_effect", "within_noise_floor", "measurable"] = "no_effect"
    elif abs(mean_delta) < noise_floor:
        verdict = "within_noise_floor"
    else:
        verdict = "measurable"
    return PairedDelta(
        mean_delta=mean_delta,
        ci_low=low,
        ci_high=high,
        n_pairs=len(values),
        n_up=n_up,
        n_down=n_down,
        n_tied=len(values) - n_up - n_down,
        sign_test_p=sign_test_p(n_up, n_down) if n_up + n_down else None,
        within_noise_floor=abs(mean_delta) < noise_floor,
        noise_floor=noise_floor,
        verdict=verdict,
        scenario_deltas=deltas,
    )


def sign_test_p(n_up: int, n_down: int) -> float:
    """Exact two-sided binomial sign test p-value over the scenarios that moved.

    Minimum-likelihood two-sided rule (the same one scipy's binomtest uses): sum the
    probability of every outcome no more likely than the observed one under p=0.5. Ties are
    excluded before calling, per the standard sign-test treatment.

    Raises:
        ValueError: when nothing moved; the caller decides what no movement means (see
            `paired_delta`, which reports None), a p-value would misrepresent it.
    """
    n = n_up + n_down
    if n == 0:
        raise ValueError(
            "sign_test_p needs at least one scenario that moved; with zero movers there is no "
            "test to run, report the absence of movement instead of a p-value"
        )
    observed = comb(n, min(n_up, n_down))
    total = sum(comb(n, k) for k in range(n + 1) if comb(n, k) <= observed)
    return total / 2**n


def sign_agreement(
    a_deltas: Mapping[str, float],
    b_deltas: Mapping[str, float],
) -> SignAgreement:
    """Per-scenario paired sign agreement between two legs' deltas (e.g. WM leg vs real leg).

    The primary sim-to-real statistic (GEV exec-bench style): over scenarios present in both
    maps, how often do the two legs agree on the DIRECTION of the effect? Pairs where either
    side is a tie are excluded and counted, not folded into agreement.

    Raises:
        ValueError: when the two maps share no scenario.
    """
    shared = sorted(set(a_deltas) & set(b_deltas))
    if not shared:
        raise ValueError(
            "no shared scenario between the two delta maps; sign agreement is paired by "
            "scenario id, so both legs must be keyed on the same ids"
        )
    ties = sum(
        1
        for sid in shared
        if abs(a_deltas[sid]) <= TIE_EPSILON or abs(b_deltas[sid]) <= TIE_EPSILON
    )
    decided = [
        sid
        for sid in shared
        if abs(a_deltas[sid]) > TIE_EPSILON and abs(b_deltas[sid]) > TIE_EPSILON
    ]
    agree = sum(1 for sid in decided if (a_deltas[sid] > 0) == (b_deltas[sid] > 0))
    return SignAgreement(
        agree=agree,
        compared=len(decided),
        ties_excluded=ties,
        fraction=agree / len(decided) if decided else None,
    )


def spearman_model_means(pairs: Sequence[tuple[float, float]]) -> DescriptiveSpearman:
    """Spearman rank correlation over model means, self-labeled as descriptive only.

    Average ranks for ties, Pearson over the ranks. The caveat is part of the return value
    because the binding sim-to-real amendment (joint-tau master, 2026-07-27) bans headlining
    this number: at n=4-11 models the null SD of a rank correlation is roughly 0.4-0.5.

    Raises:
        ValueError: with fewer than 3 pairs, where a rank correlation is vacuous.
    """
    if len(pairs) < 3:
        raise ValueError(
            f"spearman over {len(pairs)} pairs is vacuous; supply at least 3 model means, and "
            f"remember the result is descriptive only at any n this program reaches"
        )
    xs = _average_ranks([p[0] for p in pairs])
    ys = _average_ranks([p[1] for p in pairs])
    rho = _pearson(xs, ys)
    return DescriptiveSpearman(
        rho=rho,
        n=len(pairs),
        caveat=(
            f"descriptive only: a model-mean rank correlation over n={len(pairs)} models has a "
            f"null SD of roughly 0.4-0.5, so this value cannot support a transfer claim on its "
            f"own; per-scenario paired sign agreement is the primary evidence"
        ),
    )


def _scenario_means(rewards_by_scenario: Mapping[str, Sequence[float]]) -> dict[str, float]:
    """Mean reward per scenario, rejecting shapes that would silently skew an aggregate."""
    if not rewards_by_scenario:
        raise ValueError(
            "no scenarios to aggregate; load scored rows first (an empty map usually means the "
            "arm's matrix has not landed yet, see common/data.py loaders)"
        )
    means: dict[str, float] = {}
    for sid, rewards in rewards_by_scenario.items():
        if not rewards:
            raise ValueError(
                f"scenario {sid!r} has no rewards; pass only scored episodes (unscored episodes "
                f"are infrastructure failures and must be excluded, never counted as 0)"
            )
        means[sid] = fsum(rewards) / len(rewards)
    return means


def _bootstrap_ci(
    values: Sequence[float], *, resamples: int, seed: int, confidence: float
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean, resampling the given cluster-level values."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    n = len(values)
    if n == 1:
        return (values[0], values[0])
    rng = Random(seed)
    means = sorted(
        fsum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)
    )
    tail = (1.0 - confidence) / 2.0
    low_index = int(tail * (resamples - 1))
    high_index = int((1.0 - tail) * (resamples - 1))
    return (means[low_index], means[high_index])


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Ranks starting at 1, ties receiving the average of the positions they span."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation; 0.0 when either side is constant (no direction to correlate)."""
    n = len(xs)
    mean_x = fsum(xs) / n
    mean_y = fsum(ys) / n
    cov = fsum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    var_x = fsum((x - mean_x) ** 2 for x in xs)
    var_y = fsum((y - mean_y) ** 2 for y in ys)
    if var_x == 0.0 or var_y == 0.0:
        return 0.0
    return cov / (var_x * var_y) ** 0.5
