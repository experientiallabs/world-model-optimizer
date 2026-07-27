"""OOD splits for routing experiments: hold out whole regions, not random scenarios.

An IID split (`split_scenario_ids`) measures interpolation; routers in production also face
drift, new tasks, and query types absent from fit. These splits measure that regime:

- `split_holdout_clusters`: embed every scenario, k-means into coarse groups, and hold out
  WHOLE groups until `test_fraction` of scenarios sit in test. The fitter never sees any
  scenario near the test queries. Works on any matrix (no id-prefix requirement); this is the
  generic analogue of ProxRouter's Leave-Task-Out regime (arXiv 2510.09852, section 4).
- `split_holdout_tasks`: hold out whole id-prefix groups (eval names). Only defined for
  matrices whose scenario ids carry `prefix:` task names; this IS Leave-Task-Out.

Both rotate what is held out via `seed` (which groups go to test), keep both sides non-empty,
and return disjoint sorted id lists, same contract as `split_scenario_ids`.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer

if TYPE_CHECKING:
    from wmo.optimize.outcomes import OutcomeMatrix
    from wmo.optimize.policy import EmbedderSpec

logger = logging.getLogger(__name__)


def split_holdout_clusters(
    matrix: OutcomeMatrix,
    *,
    embedder: EmbedderSpec,
    test_fraction: float = 0.3,
    n_clusters: int | None = None,
    seed: int = 0,
    kmeans_seed: int = 1234,
) -> tuple[list[str], list[str]]:
    """Hold out whole embedding clusters as the test side (see module docstring).

    The clustering itself is pinned to `kmeans_seed` so seeds vary WHICH clusters are held
    out, not the cluster geometry; that keeps the 5-seed spread about drift exposure, not
    k-means jitter.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    scenario_tasks: dict[str, str] = {}
    for outcome in matrix.outcomes:
        scenario_tasks.setdefault(outcome.scenario_id, outcome.task)
    scenario_ids = list(scenario_tasks)
    if len(scenario_ids) < 4:
        raise ValueError(f"need at least 4 scenarios to split, got {len(scenario_ids)}")
    k = n_clusters or max(4, min(16, len(scenario_ids) // 8))
    k = min(k, len(scenario_ids))

    embeddings = np.asarray(embedder.build().embed([scenario_tasks[sid] for sid in scenario_ids]))
    embeddings = Normalizer(norm="l2").fit_transform(embeddings)
    labels = KMeans(n_clusters=k, random_state=kmeans_seed, n_init="auto").fit_predict(embeddings)
    groups: dict[int, list[str]] = {}
    for sid, label in zip(scenario_ids, labels, strict=True):
        groups.setdefault(int(label), []).append(sid)
    return _holdout_groups(groups, test_fraction=test_fraction, seed=seed)


def split_holdout_tasks(
    matrix: OutcomeMatrix, *, test_fraction: float = 0.3, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Hold out whole id-prefix task groups (Leave-Task-Out). Needs `prefix:` ids."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    groups: dict[str, list[str]] = {}
    for sid in matrix.scenario_ids():
        if ":" not in sid:
            raise ValueError(f"scenario id '{sid}' has no task prefix; use holdout-clusters")
        groups.setdefault(sid.split(":", 1)[0], []).append(sid)
    if len(groups) < 2:
        raise ValueError("need at least 2 task prefixes to hold tasks out")
    return _holdout_groups(groups, test_fraction=test_fraction, seed=seed)


def _holdout_groups[GroupKey: (int, str)](
    groups: dict[GroupKey, list[str]],
    *,
    test_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Greedily assign whole groups to test until `test_fraction` of scenarios are there."""
    total = sum(len(ids) for ids in groups.values())
    keys = sorted(groups, key=str)
    random.Random(seed).shuffle(keys)
    test: list[str] = []
    taken = []
    for key in keys:
        if len(test) >= test_fraction * total or len(taken) == len(keys) - 1:
            break
        test.extend(groups[key])
        taken.append(key)
    fit = [sid for key in keys if key not in set(taken) for sid in groups[key]]
    if not fit or not test:
        raise ValueError("degenerate holdout split; check group sizes vs test_fraction")
    logger.info(
        "holdout split: %d/%d groups -> test (%d/%d scenarios, target %.0f%%)",
        len(taken),
        len(keys),
        len(test),
        total,
        100 * test_fraction,
    )
    return sorted(fit), sorted(test)
