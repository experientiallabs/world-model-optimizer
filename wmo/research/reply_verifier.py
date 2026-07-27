"""Distilled reply verifier: a cheap linear scorer over reply embeddings, standing in for the judge.

The Zooter path (arXiv 2311.08692, NAACL 2024). The expensive pinned judge already scored every
stored rollout, so its verdicts are free supervision; the question is whether a cheap scorer can
learn enough from the reply TEXT to pick the better of two rollouts of the same scenario. That
selection is the only thing a best-of-n router needs, and free features
(`wmo.research.posthoc_bounds`) reach only ~66% correct on decisive cells, harvesting a fraction
of the oracle-of-2 gap.

TWO HEADS, and the contrast is the point:

- `absolute` is the naive form: regress reply embedding -> reward. It is what "distill the judge"
  literally means, and it is expected to underperform at SELECTION, because most of the variance
  in reward is between-scenario difficulty (hard scenarios score low whatever the rollout did),
  and a scorer that spends its capacity on difficulty carries no within-cell signal. This is the
  same between/within confound that made the pooled-sign free selector anti-correlated.
- `pairwise` optimizes the actual task: for cells whose two episodes got different rewards,
  regress the embedding DIFFERENCE onto the reward difference. Difficulty cancels in the
  subtraction, so the fitted direction can only encode rollout quality at fixed difficulty.

Both are closed-form ridge (numpy only, no torch), so a fit is deterministic and instant.

Information boundary: the scorer reads reply text only. Reward labels are used at FIT time on the
fit split and never at selection time, so a fitted verifier is a deployable selector, not an
oracle. Any positive result must be checked against `shuffled` labels, which have to collapse to
chance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from wmo.optimize.outcomes import ScenarioOutcome

EpisodeKey = tuple[str, str, int]


def episode_key(outcome: ScenarioOutcome) -> EpisodeKey:
    """(scenario, model, episode) - identifies one rollout, which is one embedding row."""
    return (outcome.scenario_id, outcome.model, outcome.episode)


class Projection(BaseModel):
    """A PCA basis fitted on FIT-SPLIT embeddings only, applied before the linear head.

    The pairwise design has roughly 600 rows against 3072 embedding dimensions, so the head is
    badly underdetermined and ridge has to shrink almost everything away. Projecting onto the top
    components first gives the same signal far fewer parameters to spend. Fitted on fit-split
    vectors only: deriving the basis from all embeddings would leak test-split structure into the
    representation even though no test label is touched.
    """

    mean: list[float]
    components: list[list[float]]  # [k, dim], orthonormal rows

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        matrix = np.atleast_2d(np.asarray(vectors, dtype=float))
        centered = matrix - np.asarray(self.mean, dtype=float)
        return centered @ np.asarray(self.components, dtype=float).T

    def apply_difference(self, differences: np.ndarray) -> np.ndarray:
        """Project DIFFERENCES of vectors: the mean cancels in a - b, so it must not be subtracted.

        Equivalent to `apply(a) - apply(b)`, which is what keeps a projected pairwise fit
        consistent with `score`, and subtracting the mean here would corrupt that identity.
        """
        matrix = np.atleast_2d(np.asarray(differences, dtype=float))
        return matrix @ np.asarray(self.components, dtype=float).T


def fit_projection(vectors: np.ndarray, components: int) -> Projection:
    """Top-`components` PCA basis by SVD (numpy only)."""
    matrix = np.asarray(vectors, dtype=float)
    mean = matrix.mean(axis=0)
    _u, _s, vt = np.linalg.svd(matrix - mean, full_matrices=False)
    keep = min(components, vt.shape[0])
    return Projection(mean=mean.tolist(), components=vt[:keep].tolist())


class ReplyVerifier(BaseModel):
    """A fitted linear scorer over reply embeddings. Higher score = predicted better rollout."""

    mode: Literal["absolute", "pairwise"]
    coef: list[float]
    intercept: float = 0.0
    alpha: float
    train_rows: int
    projection: Projection | None = None

    def score(self, vectors: np.ndarray) -> np.ndarray:
        """Score one or many embedding rows, projecting first when the head was fitted that way."""
        matrix = np.atleast_2d(np.asarray(vectors, dtype=float))
        if self.projection is not None:
            matrix = self.projection.apply(matrix)
        return matrix @ np.asarray(self.coef, dtype=float) + self.intercept


def _ridge(features: np.ndarray, targets: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Closed-form ridge with an unpenalized intercept (centered solve)."""
    mean_x = features.mean(axis=0)
    mean_y = float(targets.mean())
    centered = features - mean_x
    gram = centered.T @ centered + alpha * np.eye(features.shape[1])
    coef = np.linalg.solve(gram, centered.T @ (targets - mean_y))
    return coef, mean_y - float(mean_x @ coef)


def fit_absolute(
    features: np.ndarray,
    rewards: np.ndarray,
    *,
    alpha: float,
    projection: Projection | None = None,
) -> ReplyVerifier:
    """Regress reply embedding -> reward (the literal judge-distillation head)."""
    design = np.asarray(features, dtype=float)
    if projection is not None:
        design = projection.apply(design)
    coef, intercept = _ridge(design, np.asarray(rewards, dtype=float), alpha)
    return ReplyVerifier(
        mode="absolute",
        coef=coef.tolist(),
        intercept=intercept,
        alpha=alpha,
        train_rows=len(design),
        projection=projection,
    )


def pairwise_design(
    cells: Sequence[Sequence[ScenarioOutcome]],
    vectors: dict[EpisodeKey, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Build (embedding differences, reward differences) over cells whose rewards disagree.

    One row per ordered pair, both directions included so the design is sign-symmetric and the
    fitted direction cannot absorb an offset. Cells whose episodes tie carry no ranking
    information and are dropped.
    """
    rows: list[np.ndarray] = []
    gaps: list[float] = []
    for episodes in cells:
        scored = [o for o in episodes if o.reward is not None and episode_key(o) in vectors]
        for i, first in enumerate(scored):
            for second in scored[i + 1 :]:
                assert first.reward is not None and second.reward is not None
                gap = first.reward - second.reward
                if gap == 0.0:
                    continue
                difference = vectors[episode_key(first)] - vectors[episode_key(second)]
                rows.extend((difference, -difference))
                gaps.extend((gap, -gap))
    if not rows:
        return np.zeros((0, 0)), np.zeros(0)
    return np.asarray(rows, dtype=float), np.asarray(gaps, dtype=float)


def fit_pairwise(
    differences: np.ndarray,
    gaps: np.ndarray,
    *,
    alpha: float,
    projection: Projection | None = None,
) -> ReplyVerifier:
    """Regress embedding difference -> reward difference; scenario difficulty cancels out.

    No intercept: the design is sign-symmetric by construction, so a constant term is exactly zero
    and fitting one would only add variance. `differences` are always in RAW embedding space; the
    projection (if any) is applied here so `score` and the fit stay in the same basis.
    """
    features = np.asarray(differences, dtype=float)
    if projection is not None:
        features = projection.apply_difference(features)
    gram = features.T @ features + alpha * np.eye(features.shape[1])
    coef = np.linalg.solve(gram, features.T @ np.asarray(gaps, dtype=float))
    return ReplyVerifier(
        mode="pairwise",
        coef=coef.tolist(),
        intercept=0.0,
        alpha=alpha,
        train_rows=len(features),
        projection=projection,
    )


def verifier_selector(
    verifier: ReplyVerifier, vectors: dict[EpisodeKey, np.ndarray]
) -> Callable[[ScenarioOutcome], tuple[float, ...]]:
    """Adapt a verifier into a `posthoc_bounds` selector key (which ranks LOWEST-first).

    Episodes with no embedding score 0.0, i.e. they neither win nor lose on a missing reply.
    """
    scores = {key: float(verifier.score(vector)[0]) for key, vector in vectors.items()}

    def key(outcome: ScenarioOutcome) -> tuple[float, ...]:
        return (-scores.get(episode_key(outcome), 0.0),)

    return key


def scenario_folds(scenario_ids: Sequence[str], folds: int, seed: int) -> list[list[str]]:
    """Split scenarios (never episodes) into folds, so no cell straddles a fold boundary.

    Folding by episode would put one rollout of a cell in train and its sibling in validation,
    which for the pairwise head is direct leakage of the very comparison being scored.
    """
    ordered = sorted(set(scenario_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(ordered)  # type: ignore[arg-type]
    return [ordered[index::folds] for index in range(folds)]


def shuffled_rewards(rewards: np.ndarray, seed: int) -> np.ndarray:
    """Permute rewards, breaking the reply->reward link. The control every positive must pass."""
    rng = np.random.default_rng(seed)
    permuted = np.array(rewards, dtype=float)
    rng.shuffle(permuted)
    return permuted
