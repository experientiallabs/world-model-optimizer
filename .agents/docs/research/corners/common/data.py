"""Read-only loaders for the corner analyses' data sources. Never regenerates anything.

The charter pins the sources: grid arm matrices under the MAIN checkout's
`.wmo/jt/grid-c2/<arm>/matrix.json` (the LIVE cohort, relaunched 2026-07-27 at tip f1ebaca6
with #330's concurrent per-cell persistence; the earlier `.wmo/jt/grid` dir is a RETIRED
cohort and is never read as evidence), the cycle-1 per-task per-arm rows
(`episode-rows.jsonl`, 180 rows), and the D-DIAL anchors (imported straight from
`wmo.optimize.knn`, never copied). `.wmo/` artifacts are machine-local and gitignored, so the
paths default to Silen's main checkout and are overridable via WMO_MAIN_CHECKOUT for anyone
replaying this analysis elsewhere.

A missing matrix returns None rather than raising: the three corner chats render what has
landed and NAME what is pending on the figure (the no-silent-caps rule in common/README.md),
so "not landed yet" is an expected state, not an error.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

# JsonObject is a runtime import (not TYPE_CHECKING) because ArmSnapshot uses it as a pydantic
# field annotation, which pydantic must resolve when the model class is built.
from wmo.core.types import JsonObject
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.sweep_partial import PARTIAL_SUFFIX, PartialHeader, read_partial
from wmo.providers.pool import load_pool

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

logger = logging.getLogger(__name__)

# The grid's three compression arms (the consumer contract; see the 2026-07-27 grid entries in
# the plan ledger). identity = uncompressed control, truncate = ratio-matched dumb control at
# llmlingua2's achieved keep 0.5656 (aggressiveness 0.33), llmlingua2-endpoint = the learned
# compressor on box-6.
GRID_ARMS: tuple[str, ...] = ("identity", "truncate", "llmlingua2-endpoint")
IDENTITY_ARM = "identity"

# Cycle-1's three measured arms (real tau2 episodes, 20 pinned holdout tasks, k=3).
CYCLE1_ARMS: tuple[str, ...] = ("teacher", "student-before", "student-after")

# Judge naming for provenance labels. The grid's WM leg scores with the world model's own
# verifier; cycle-1 rows score with tau2's reward, which is not purely deterministic (7 of the
# 20 holdout tasks include tau2's NL-assertion judge in their reward basis, counted from the
# rows' reward_basis field).
CYCLE1_JUDGE = "tau2 reward (7/20 holdout tasks include tau2's NL-assertion judge)"


class Cycle1Row(BaseModel):
    """One row of cycle-1's episode-rows.jsonl: one episode of one arm on one holdout task."""

    arm: str
    episode: str
    task_id: str
    attempt: int
    infra_failed: bool
    reward: float | None
    passed: bool
    termination_reason: str
    reward_basis: list[str]
    duration_s: float
    messages: int


def main_checkout() -> Path:
    """The main world-model-harness checkout holding the machine-local `.wmo` artifacts."""
    return Path(
        os.environ.get("WMO_MAIN_CHECKOUT", "~/Desktop/Projects/world-model-harness")
    ).expanduser()


def grid_dir() -> Path:
    """The LIVE canonical tau grid cohort (pins, ledger, per-arm matrices).

    grid-c2 replaced the original `jt/grid` cohort on 2026-07-27 (DECISIONS: relaunched at
    tip f1ebaca6 with concurrent sweep persistence after the original persisted zero cells
    in ~1.5h). The old dir still exists on disk; it is retired evidence and nothing here may
    read it.
    """
    return main_checkout() / ".wmo" / "jt" / "grid-c2"


def cycle1_run_dir() -> Path:
    """Cycle-1's distill run directory (rows, gate, evals, spend)."""
    return main_checkout() / ".wmo" / "distill-runs" / "tau2-cycle1"


def load_arm_matrix(arm: str, *, root: Path | None = None) -> OutcomeMatrix | None:
    """One grid arm's merged matrix, or None (with a logged reason) while it has not landed."""
    if arm not in GRID_ARMS:
        raise ValueError(f"unknown grid arm {arm!r}; the cohort's arms are {GRID_ARMS}")
    path = (root or grid_dir()) / arm / "matrix.json"
    if not path.exists():
        logger.info("grid arm %r has no merged matrix yet at %s", arm, path)
        return None
    return OutcomeMatrix.model_validate_json(path.read_text(encoding="utf-8"))


def load_arm_meta(arm: str, *, root: Path | None = None) -> JsonObject | None:
    """One grid arm's merge metadata (cohort pins, measured compression), or None."""
    if arm not in GRID_ARMS:
        raise ValueError(f"unknown grid arm {arm!r}; the cohort's arms are {GRID_ARMS}")
    path = (root or grid_dir()) / arm / "matrix.meta.json"
    if not path.exists():
        return None
    loaded: JsonObject = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def load_cycle1_rows(*, run_dir: Path | None = None) -> list[Cycle1Row]:
    """Cycle-1's 180 per-task per-arm rows (real tau2 episodes on the 20 pinned holdout tasks).

    Raises:
        FileNotFoundError: when the run dir has no episode-rows.jsonl; unlike a pending grid
            arm this file already exists, so its absence means a wrong path, not patience.
    """
    path = (run_dir or cycle1_run_dir()) / "episode-rows.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"no episode-rows.jsonl at {path}; cycle-1's rows were delivered to the run dir on "
            f"2026-07-27, so point run_dir (or WMO_MAIN_CHECKOUT) at the main checkout"
        )
    with path.open(encoding="utf-8") as handle:
        return [Cycle1Row.model_validate_json(line) for line in handle if line.strip()]


class ArmSnapshot(BaseModel):
    """One grid arm's rows plus how complete they are.

    `status` travels onto every figure footnote: a chart rendered from two of four chunks
    must say so, because the runner's scenario cut is deterministic and a partial read is a
    BIASED subset (earlier scenarios only), not a random sample. Pre-retry chunk reads can
    only overcount unscored cells, never invent scored ones (the retry pass re-executes
    transient failures individually before the merge).
    """

    name: str
    matrix: OutcomeMatrix
    status: str  # "merged" | "partial (k chunk file(s), pre-retry)"
    meta: JsonObject | None = None


def load_arm_snapshot(arm: str, *, root: Path | None = None) -> ArmSnapshot | None:
    """This arm's merged matrix, else its pre-retry chunk files concatenated, else None.

    The chunk path exists because the master's grid-timing entry (2026-07-27) has merged
    matrices landing ~12-16h after relaunch while per-chunk matrices appear along the way:
    corner pipelines render what has landed and label completeness. Chunk pool snapshots
    must agree (the runner's merge enforces full equality; this loader checks the names row
    selection keys on).
    """
    matrix = load_arm_matrix(arm, root=root)
    if matrix is not None:
        return ArmSnapshot(
            name=arm, matrix=matrix, status="merged", meta=load_arm_meta(arm, root=root)
        )
    arm_dir = (root or grid_dir()) / arm
    chunks = sorted(
        arm_dir.glob("chunk-*.json"), key=lambda p: int(p.stem.removeprefix("chunk-"))
    )
    matrices = [
        OutcomeMatrix.model_validate_json(path.read_text(encoding="utf-8")) for path in chunks
    ]
    # grid-c2 (#330) persists every cell to a `chunk-N.json.partial.jsonl` sidecar as it
    # completes; a chunk whose matrix has not been written yet still has live evidence there.
    # The product's own parser does the reading (header check, torn-final-line tolerance,
    # plan-identity guard against a stale sidecar from another cohort); rows are priced with
    # the cohort's pool.toml, since a sidecar carries only the pool digest.
    sidecars = [
        path
        for path in sorted(arm_dir.glob("chunk-*.json.partial.jsonl"))
        if not path.with_name(path.name.removesuffix(PARTIAL_SUFFIX)).exists()
    ]
    sidecar_rows: list[ScenarioOutcome] = []
    for path in sidecars:
        first_line = path.read_text(encoding="utf-8").split("\n", 1)[0]
        header = PartialHeader.model_validate_json(first_line)
        sidecar_rows.extend(read_partial(path, header.identity))
    if not chunks and not sidecar_rows:
        logger.info("grid arm %r has no chunk files or sidecars yet under %s", arm, arm_dir)
        return None
    if len({tuple(m.model_names()) for m in matrices}) > 1:
        raise ValueError(f"{arm}: chunk files disagree on the pool; refusing to concatenate")
    pool = (
        matrices[0].pool
        if matrices
        else load_pool((root or grid_dir()) / "pool.toml").models
    )
    parts = []
    if chunks:
        parts.append(f"{len(chunks)} chunk file(s)")
    if sidecars:
        parts.append(f"{len(sidecars)} live sidecar(s), {len(sidecar_rows)} cell(s)")
    return ArmSnapshot(
        name=arm,
        matrix=OutcomeMatrix(
            pool=pool,
            outcomes=[o for m in matrices for o in m.outcomes] + sidecar_rows,
        ),
        status=f"partial ({', '.join(parts)}, pre-retry)",
    )


def all_arm_snapshots(*, root: Path | None = None) -> list[ArmSnapshot]:
    """Every canonical arm with any data on disk, in the consumer contract's order."""
    return [
        snapshot
        for arm in GRID_ARMS
        if (snapshot := load_arm_snapshot(arm, root=root)) is not None
    ]


def rewards_by_scenario(
    outcomes: Iterable[ScenarioOutcome], *, model: str
) -> dict[str, list[float]]:
    """One pool model's scored rewards grouped per scenario, ready for common/stats.

    Unscored rows are excluded (an infrastructure failure is not a judge verdict of 0), which
    matches the scorecard's aggregation exactly.
    """
    grouped: dict[str, list[float]] = {}
    for outcome in outcomes:
        if outcome.model == model and outcome.reward is not None:
            grouped.setdefault(outcome.scenario_id, []).append(outcome.reward)
    return grouped


def cycle1_rewards_by_task(rows: Sequence[Cycle1Row], *, arm: str) -> dict[str, list[float]]:
    """One cycle-1 arm's scored rewards grouped per holdout task, ready for common/stats."""
    if arm not in CYCLE1_ARMS:
        raise ValueError(f"unknown cycle-1 arm {arm!r}; the measured arms are {CYCLE1_ARMS}")
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row.arm == arm and not row.infra_failed and row.reward is not None:
            grouped.setdefault(row.task_id, []).append(row.reward)
    return grouped
