"""The routing sweep's library core: plan a closed-loop measurement, then run it.

`wmo optimize route sweep` and the sweep stage of `wmo optimize model` are two faces of THIS
module, so the two cannot drift. What lives here once: the scenario cut (the corpus's held-out
band, sorted by trace id, capped by `scenarios`), the backend pre-flight that resolves every
candidate before anything is spent, the projected cost table's arithmetic, and the coverage
contract that decides whether the resulting matrix is fit-ready.

Rendering deliberately does NOT live here. Every function returns typed data or raises
`SweepError` carrying a message its caller prints, so each CLI face owns its own console, its own
spend confirmation, and its own progress display while the measurement itself is single-sourced.

The two seams a caller supplies are the world model and the env factory. They are parameters
rather than imports so the caller decides how the model is loaded and how the env is
constructed, and so tests can stub both at the CLI module they already patch.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from wmo.config import ArtifactPaths, HarnessConfig, load_config
from wmo.core.types import Action, EnvState, Observation
from wmo.engine import split_holdout
from wmo.env.closed_loop import (
    CellKey,
    PoolCell,
    ScoringEnv,
    pool_cells,
    run_cells,
    scenario_id,
)
from wmo.env.llm_agent import DEFAULT_HISTORY_CHARS
from wmo.env.scenarios import Scenario, scenarios_from_traces, tools_hint_from_traces
from wmo.ingest import get_adapter
from wmo.optimize.compression import CompressionConfig, compression_signature
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.reward import EpisodeScore
from wmo.optimize.sweep_partial import (
    PartialSweepError,
    PartialWriter,
    PlanIdentity,
    partial_path,
    read_partial,
)
from wmo.providers.base import ProviderKind, TokenUsage
from wmo.providers.pool import ModelPool, load_pool, prepare_pool_provider
from wmo.serving.traces_source import TRACES_FILENAME, local_traces_path
from wmo.tracking import RunRecord, merge_run_records, save_run

if TYPE_CHECKING:
    from wmo.core.types import Trace
    from wmo.engine.world_model import WorldModel
    from wmo.env.base import Env

logger = logging.getLogger(__name__)


class SweepError(ValueError):
    """A sweep cannot proceed, for a reason the operator can fix.

    Every message names what went wrong and what to do about it, because both CLI faces render
    it verbatim as a usage error (AGENTS rule 9: error messages are part of the interface).
    """


# What each kind still cannot know before its first cell, and why. Both are measured properties
# of the backend (see `BedrockProvider.prepare` and `TinkerChatProvider.prepare`), not caution: a
# pre-flight that resolved them would have to make a request, which is the one thing it may not do.
DEFERRED_RISK: dict[ProviderKind, str] = {
    ProviderKind.BEDROCK: (
        "AWS credentials (boto3 resolves them by walking a chain that can reach the "
        "instance-metadata endpoint over the network, and it builds a client with no credentials "
        "at all)"
    ),
    ProviderKind.TINKER: (
        "the Tinker service being reachable and serving this model (constructing the client "
        "connects and pins a server-side session for the whole process)"
    ),
}


class DeferredRisk(BaseModel):
    """One candidate's residual first-cell risk that a request-free pre-flight cannot close."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate: str
    kind: ProviderKind
    risk: str


class PoolPreflight(BaseModel):
    """The candidate roster, proven constructible, plus what could still fail at a first cell."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pool: ModelPool
    deferred: tuple[DeferredRisk, ...]


class CostLine(BaseModel):
    """One candidate's projected sweep spend under the stated per-call token assumption."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate: str
    episodes: int
    calls: int
    input_per_mtok: float
    output_per_mtok: float
    usd: float


class SweepPlan(BaseModel):
    """Everything a sweep will do, computed before a single cell is paid for.

    Held as data so the same plan can be printed as a cost table by `route sweep`, folded into
    `optimize model`'s one plan table, and then executed, without any of the three recomputing
    what the others decided.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_dir: Path
    out_path: Path
    pool: ModelPool
    scenarios: tuple[Scenario, ...]
    episodes: int
    max_steps: int
    tools_hint: str | None
    # How much of each observation the agent sees on later turns. Part of the plan because it
    # changes what the candidates are measured on, so two matrices swept at different values are
    # not comparable and the value has to travel with the run that produced them.
    history_chars: int = DEFAULT_HISTORY_CHARS
    # The D-COMPRESS arm this sweep measures. Part of what the plan IS, not a detail of how it
    # runs: it decides what the rewards MEAN, so two matrices swept under different compressors
    # are different evidence, and a resumed run whose compressor changed must re-measure.
    compression: CompressionConfig | None = None
    # How many cells run at once. NOT part of the plan's identity: it changes how long the
    # measurement takes, never what a cell measures, so a run resumed at a different value
    # continues the same cohort rather than re-buying it. 1 is the sequential default.
    max_concurrency: int = Field(default=1, ge=1)
    trace_count: int  # traces the corpus ingested, which is what decides `tiny_corpus`
    tiny_corpus: bool  # too small for a held-out band, so the scenarios are not leak-free
    assume_input_tokens: int
    assume_output_tokens: int
    cost_lines: tuple[CostLine, ...]

    @property
    def cells(self) -> int:
        """Cells the sweep will run: candidates x scenarios x episodes."""
        return len(self.pool.models) * len(self.scenarios) * self.episodes

    @property
    def total_usd(self) -> float:
        """The projected candidate-side spend (a projection, never a measurement)."""
        return sum(line.usd for line in self.cost_lines)

    @property
    def identity(self) -> PlanIdentity:
        """What this plan MEASURES, as the value a resumed run is checked against.

        Everything a matrix's rows depend on for comparability: which candidates at which prices,
        which scenario cut, how many episodes of how many steps, how much of each observation the
        agent saw, and which compression arm. Two plans that agree here produce rows that belong
        in one matrix; two that disagree produce two arms, and merging them would be a fabricated
        comparison. `identity.digest` is the short form for stamping into logs and artifacts.
        """
        return PlanIdentity(
            pool=_pool_digest(self.pool),
            scenarios=tuple(scenario_id(scenario) for scenario in self.scenarios),
            episodes=self.episodes,
            max_steps=self.max_steps,
            history_chars=self.history_chars,
            compression=compression_signature(self.compression),
        )


def resolve_config(model_dir: Path) -> HarnessConfig:
    """The built model's config, or a `SweepError` naming why it could not be read."""
    try:
        return load_config(model_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise SweepError(str(exc)) from exc


def preflight_pool(pool_file: Path) -> PoolPreflight:
    """Load the candidate pool and resolve every backend locally, BEFORE anything is spent.

    `evaluate_pool` builds a candidate's provider lazily at that candidate's FIRST CELL, and every
    backend then builds its SDK client lazily inside that first call, so any reason a candidate
    cannot be used (an unset `api_key_env`, a kind that refuses the explicit key the entry names,
    a missing SDK extra, an Azure config with no api_version, a Bedrock entry whose region
    resolves nowhere) used to abort the run as a raw traceback after every earlier candidate had
    been fully paid for, with no matrix written. `prepare_pool_provider` closes that: it checks the
    kind's static requirements from the entry alone, then forces the lazy client to be BUILT, which
    imports the SDK and resolves credentials from the environment and local credential files.

    No request, ever. Every `prepare` is documented network-free, and verifying a candidate over
    the wire is deliberately NOT done here: `wmo providers verify` bills a real call per model,
    which would spend money inside a pre-flight whose whole job is to run before any spend is
    authorized, and would make the cost estimate printed next understate what the command had
    already spent. Some backends therefore keep a residual gap (`DEFERRED_RISK`), returned per
    entry rather than left for the operator to discover mid-sweep.

    Every prepared provider is discarded: `evaluate_pool` still builds its own per cell, so
    per-cell provider state (the tinker provider's per-episode prompt history) is unchanged.

    Reports EVERY unusable candidate, not just the first: a pool is edited as a file, so an
    operator fixing one entry at a time pays a full round trip per typo.

    Only ENABLED entries are candidates: `enabled = false` is the roster's per-model toggle, so
    a turned-off entry is neither prepared nor swept (and is not an error even when its backend
    is unusable, which is half the point of turning one off). A pool with every entry off is
    answered like an empty one, naming the flag to flip.

    Raises:
        SweepError: The pool could not be read, every candidate is disabled, or one or more
            enabled candidates cannot be used.
    """
    try:
        full_roster = load_pool(pool_file)
    except (FileNotFoundError, ValueError) as exc:
        raise SweepError(str(exc)) from exc
    enabled = full_roster.enabled_models()
    if not enabled:
        raise SweepError(
            f"every candidate in {pool_file} is disabled (enabled = false); re-enable at least "
            "one [[model]] entry to have a pool to measure"
        )
    pool = ModelPool(models=enabled)
    problems: list[str] = []
    for entry in pool.models:
        try:
            prepare_pool_provider(entry)
        except Exception as exc:  # noqa: BLE001 - anything here is a usage error, never a spend
            # `prepare_pool_provider` already prefixes its own failures with the entry name and
            # kind; a surprise from deeper down gets the same identification so the file is
            # editable from the message.
            detail = str(exc)
            problems.append(
                detail
                if detail.startswith(f"pool model '{entry.name}'")
                else f"pool model '{entry.name}' (kind={entry.kind.value}): {detail}"
            )
    if problems:
        raise SweepError(
            "; ".join(problems)
            + ". Fix or remove those entries in the pool file, then re-run (checked all "
            + f"{len(pool.models)} candidate(s) before spending anything)"
        )
    return PoolPreflight(
        pool=pool,
        deferred=tuple(
            DeferredRisk(candidate=entry.name, kind=entry.kind, risk=DEFERRED_RISK[entry.kind])
            for entry in pool.models
            if entry.kind in DEFERRED_RISK
        ),
    )


def plan_sweep(
    *,
    model_dir: Path,
    config: HarnessConfig,
    pool: ModelPool,
    out_path: Path,
    traces_file: Path | None,
    scenarios: int,
    episodes: int,
    max_steps: int,
    assume_input_tokens: int,
    assume_output_tokens: int,
    history_chars: int = DEFAULT_HISTORY_CHARS,
    compression: CompressionConfig | None = None,
    max_concurrency: int = 1,
) -> SweepPlan:
    """Cut the held-out scenario set and project the spend, without touching the filesystem.

    Everything knowable without spending is settled here: an `out_path` that cannot be written
    and a corpus that carries no task prompt are both boundary errors, raised before a caller
    asks its operator to authorize anything.

    `max_concurrency` is how many cells the sweep runs at once (1 = sequential). It is on the
    PLAN because an operator confirms the plan, and how hard a run leans on a provider's rate
    limit is something they should see before they say yes; it is not part of the plan's identity,
    because it changes nothing about what a cell measures.

    Raises:
        SweepError: The destination is unwritable, the corpus is missing or unreadable, the
            held-out band carries no task prompt to measure, or `max_concurrency` is below 1.
    """
    if max_concurrency < 1:
        raise SweepError(
            f"--concurrency must be at least 1, got {max_concurrency}: 1 runs the cells one at "
            "a time, higher runs that many at once"
        )
    _check_out_writable(out_path)
    traces = _corpus_traces(model_dir, config.trace_adapter, traces_file)
    train, holdout, tiny_corpus = split_holdout(
        traces, config.train_split, (1.0 - config.train_split) / 2
    )
    # Sorted by trace id first, so `scenarios` always cuts the same prefix of the same corpus.
    ordered = sorted(holdout, key=lambda trace: trace.trace_id)
    cut = scenarios_from_traces(ordered)[:scenarios]
    if not cut:
        raise SweepError(
            f"the {len(holdout)} held-out trace(s) of world model '{model_dir.name}' carry no "
            "task prompt, so there is nothing to measure; rebuild from a corpus whose traces "
            "record the instruction they were given"
        )
    return SweepPlan(
        model_dir=model_dir,
        out_path=out_path,
        pool=pool,
        scenarios=tuple(cut),
        episodes=episodes,
        max_steps=max_steps,
        tools_hint=tools_hint_from_traces(train) or None,
        history_chars=history_chars,
        compression=compression,
        max_concurrency=max_concurrency,
        trace_count=len(traces),
        tiny_corpus=tiny_corpus,
        assume_input_tokens=assume_input_tokens,
        assume_output_tokens=assume_output_tokens,
        cost_lines=tuple(
            _estimate_cost(
                pool,
                episodes_per_candidate=len(cut) * episodes,
                calls_per_episode=max_steps,
                input_tokens=assume_input_tokens,
                output_tokens=assume_output_tokens,
            )
        ),
    )


class SweepRun(BaseModel):
    """A finished sweep: the evidence, and BOTH sides of what it cost.

    The two costs are never one number. `candidate_usd` is what the pool models charged, which is
    the customer-facing serving cost the policy is fitted to trade off. `world_model_usage` is
    what the simulator itself charged to run the evaluation (its own serve steps plus the judge),
    which is eval infrastructure a customer never pays. Blending them would overstate the price of
    serving and understate the price of measuring.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    matrix: OutcomeMatrix
    # Candidate-side spend with the compressor's bill IN it. The D-COMPRESS accounting rule is
    # that every savings number is effective cost per completed task, compressor inference cost
    # and latency included, and `wmo.serving.savings` already sums both; a sweep reporting only
    # the model half would be a second, quieter answer to the same question.
    candidate_usd: float
    # The compressor's share of that total. Kept separately because it is the part the plan
    # table cannot project, so the spend forecast has to divide it back out to stay like-for-like
    # (see `wmo.optimize.pipeline.project_sweep_spend`).
    compressor_usd: float = 0.0
    world_model_usage: RunRecord
    episodes_metered: int  # episodes whose world-model session reported usage
    episodes_unmetered: int  # episodes whose env exposed none (see `metering_gap`)
    usage_path: Path | None  # where the kind="sweep" record was persisted, if it was
    # Cells this attempt did NOT run because an earlier attempt already had (see the resume
    # contract in `execute_sweep`). Zero for a sweep that measured its whole grid.
    resumed_cells: int = 0
    # Cells this attempt measured AGAIN on the caller's instruction, after an earlier attempt
    # produced a row it did not want to keep.
    remeasured_cells: int = 0

    @property
    def world_model_usd(self) -> float:
        """What the simulator itself charged to run this sweep."""
        return self.world_model_usage.total.cost_usd

    @property
    def metering_gap(self) -> str | None:
        """Why the world-model figure is not the whole sweep, or None when it covers every cell.

        A caller prints this INSTEAD of the number when nothing was metered: a $0.00 that means
        "not measured" is the kind of zero the numbers-honesty rule exists to forbid. A resumed
        run reports the same way for a different reason: its number is real, but it covers only
        the cells THIS attempt ran, and the earlier attempt persisted its own record.
        """
        if self.episodes_unmetered == 0:
            return self._resume_note
        if self.episodes_metered == 0:
            return (
                "not measured: the env this sweep ran on exposes no usage record, so the world "
                "model's own serve and judge cost is unknown rather than zero"
            )
        partial = (
            f"partial: {self.episodes_metered} of "
            f"{self.episodes_metered + self.episodes_unmetered} episode(s) reported usage"
        )
        note = self._resume_note
        return f"{partial}; {note}" if note else partial

    @property
    def _resume_note(self) -> str | None:
        """How a resume narrows what the world-model figure covers, or None when it did not."""
        if self.resumed_cells == 0:
            return None
        return (
            f"this attempt only: {self.resumed_cells} cell(s) came from the earlier attempt, "
            "whose own world-model spend is in its own run record"
        )


def execute_sweep(
    plan: SweepPlan,
    *,
    world_model: WorldModel,
    env_factory: Callable[[], Env],
    on_outcome: Callable[[ScenarioOutcome], None] | None = None,
    runs_dir: Path | None = None,
    remeasure: frozenset[CellKey] = frozenset(),
) -> SweepRun:
    """Run every cell of `plan` against a FROZEN world model, write the matrix, meter both sides.

    Frozen for the whole sweep (the `wmo.evals.closed_loop` precedent): without it a candidate's
    PREDICTED steps enter the shared retrieval buffer and become demos for the next candidate, so
    the comparison this matrix exists to make would depend on sweep order. Freezing is also what
    makes concurrent cells safe: the shared retrieval index is read-only for the whole run, so
    parallel episodes cannot race to write it (see `plan.max_concurrency`).

    `env_factory` must build an env that scores on close (`score_on_close=True`): a matrix
    without verified rewards is not evidence, and a cell refuses an env that does not score. It
    is called once per CELL, on the worker thread running that cell, so it must be safe to call
    from several threads (`WorldModelEnv` is: each env owns its own world-model session).

    NOTHING IS LOST TO A CRASH. Every cell is appended to `<out>.partial.jsonl` the moment it
    completes, and a later call with the same plan measures only the cells that file does not
    already hold. A sidecar whose rows were measured under a different plan is refused rather
    than merged (`SweepPlan.identity`). The sidecar is removed once the matrix is written, so a
    finished sweep leaves exactly the artifact it always left.

    The world model opens one metered session per episode and `WorldModelEnv.close` leaves that
    session's final `RunRecord` on the env. Those records used to die with the env: the sweep
    said the simulator's cost was "metered separately" and then nothing anywhere persisted it, so
    the eval-infrastructure half of a sweep's bill was unaccountable. They are harvested here as
    each env closes and rolled into ONE `kind="sweep"` record under `runs_dir`, beside the build
    and serve records the same directory already holds. A RESUMED run records only what IT spent;
    the attempt that bought the earlier cells wrote its own record, so nothing is double counted
    and `metering_gap` says the figure covers this attempt alone.

    Args:
        plan: What to measure, from `plan_sweep`.
        world_model: The model to freeze and measure against.
        env_factory: Builds one scoring env per episode; called per cell, from worker threads.
        on_outcome: Per-cell progress callback, fired from the worker thread that finished the
            cell and serialized so two cells never interleave inside it.
        runs_dir: Where to persist the run record. Defaults to the model's own `runs/`; pass a
            path to redirect it, and nothing is written when the sweep metered no session at all.
        remeasure: Cells to buy again even though the sidecar already holds them, for a retry
            pass over cells an earlier attempt lost to a transient fault. Their new rows carry
            `remeasured=True` and replace the earlier ones.

    Raises:
        SweepError: The sidecar beside `out_path` belongs to a different plan, or is damaged.
    """
    resumed = _resume(plan, remeasure)
    harvest = _SessionHarvest(env_factory)
    destination = runs_dir or ArtifactPaths(plan.model_dir).runs
    measured: list[ScenarioOutcome] = []

    def persist(outcome: ScenarioOutcome) -> None:
        """Bank one measured cell durably, then let the caller's progress display have it.

        Called under the one lock `run_cells` serializes its callback with, which is why the
        writer needs no lock of its own and why the log's order is the completion order.
        """
        resumed.writer.append(outcome)
        if on_outcome is not None:
            on_outcome(outcome)

    try:
        with world_model.frozen():
            measured = run_cells(
                resumed.pending,
                harvest,
                max_steps=plan.max_steps,
                tools_hint=plan.tools_hint,
                history_chars=plan.history_chars,
                on_outcome=persist,
                compression=plan.compression,
                max_concurrency=plan.max_concurrency,
            )
    finally:
        # A cell can raise (a provider that fails to build, an env that does not score), and the
        # episodes that already ran were still paid for on the world model's side. Persisting the
        # harvest here keeps that spend accountable instead of dying with the exception; the
        # candidate-side rows survive in the sidecar, which is what the next run resumes from.
        usage, usage_path = _persist_harvest(harvest, destination)
        resumed.writer.close()
    matrix = OutcomeMatrix(pool=plan.pool.models, outcomes=_assemble(plan, resumed.rows, measured))
    matrix.save(plan.out_path)
    # Only now: while the matrix was unwritten the sidecar was the only copy of these cells.
    resumed.writer.discard()
    return SweepRun(
        matrix=matrix,
        candidate_usd=sum(
            outcome.cost_usd + outcome.compressor_cost_usd for outcome in matrix.outcomes
        ),
        compressor_usd=sum(outcome.compressor_cost_usd for outcome in matrix.outcomes),
        world_model_usage=usage,
        episodes_metered=len(harvest.records),
        episodes_unmetered=harvest.unmetered,
        usage_path=usage_path,
        resumed_cells=len(resumed.rows),
        remeasured_cells=sum(1 for cell in resumed.pending if cell.remeasured),
    )


@dataclass(frozen=True)
class _Resume:
    """What an earlier attempt at this plan left behind, and what is therefore left to run.

    A dataclass rather than a model because it carries an OPEN FILE: the sidecar writer is a live
    resource with a lifetime, not validated data to persist.
    """

    rows: dict[str, ScenarioOutcome]  # surviving row per reused cell, keyed by CellKey JSON
    pending: list[PoolCell]  # cells this attempt must measure, in canonical order
    writer: PartialWriter


def _resume(plan: SweepPlan, remeasure: frozenset[CellKey]) -> _Resume:
    """Read the sidecar, decide which cells still need buying, and open the log for appending.

    A row is reused when the sidecar holds it AND the caller did not ask for that cell again.
    Unscored rows are reused too: the cell ran, the episode was paid for, and its `error` is the
    diagnosis. Re-running it is a decision the caller makes with `remeasure`, not one a resume
    makes silently on their behalf and their budget.

    Raises:
        SweepError: The sidecar belongs to a different plan, or cannot be read.
    """
    path = partial_path(plan.out_path)
    identity = plan.identity
    try:
        rows = read_partial(path, identity)
        writer = PartialWriter(path, identity)
    except PartialSweepError as exc:
        raise SweepError(str(exc)) from exc
    except OSError as exc:
        raise SweepError(f"cannot write the partial sweep log at {path}: {exc}") from exc
    reusable: dict[str, ScenarioOutcome] = {}
    for row in rows:  # append-only log: a later row for a cell supersedes an earlier one
        key = CellKey.of(row)
        if key not in remeasure:
            reusable[key.model_dump_json()] = row
    pending: list[PoolCell] = []
    for cell in pool_cells(plan.pool, list(plan.scenarios), episodes_per_scenario=plan.episodes):
        if cell.key.model_dump_json() in reusable:
            continue
        pending.append(cell.model_copy(update={"remeasured": cell.key in remeasure}))
    if reusable:
        logger.info(
            "resuming sweep %s: %d cell(s) already measured, %d to go",
            identity.digest,
            len(reusable),
            len(pending),
        )
    return _Resume(rows=reusable, pending=pending, writer=writer)


def _assemble(
    plan: SweepPlan, reused: dict[str, ScenarioOutcome], measured: list[ScenarioOutcome]
) -> list[ScenarioOutcome]:
    """The matrix's rows in canonical cell order, whatever order they were measured in.

    Deterministic ORDER, not just deterministic content: `fit` and `tune` identify a matrix by the
    sha256 of its bytes, so a matrix whose rows follow finish time would get a different digest
    every run and every artifact stamped with it would look like new evidence. Sorting by cell
    position (rather than, say, by scenario id) also means a concurrent sweep and a sequential one
    of the same plan produce the same file, byte for byte.
    """
    fresh = {CellKey.of(outcome).model_dump_json(): outcome for outcome in measured}
    ordered: list[ScenarioOutcome] = []
    for cell in pool_cells(plan.pool, list(plan.scenarios), episodes_per_scenario=plan.episodes):
        key = cell.key.model_dump_json()
        # This attempt's row wins over a reused one: a remeasured cell is measured precisely to
        # replace what the sidecar holds. A cell in neither is one no attempt has run, which the
        # matrix simply omits (the coverage contract is what judges the hole).
        row = fresh.get(key, reused.get(key))
        if row is not None:
            ordered.append(row)
    return ordered


def resumable_cells(plan: SweepPlan) -> int:
    """How many of `plan`'s cells a previous attempt already measured and left in the sidecar.

    Called BEFORE the spend confirmation, so a resumed run can say how much of the projected cost
    it is not going to pay again, and so a sidecar that belongs to a different plan is refused
    while refusing is still free. Zero when there is nothing to resume.

    Raises:
        SweepError: The sidecar beside the plan's `out_path` belongs to a different plan, or is
            damaged. Same message the run itself would raise, delivered before the money question.
    """
    try:
        rows = read_partial(partial_path(plan.out_path), plan.identity)
    except PartialSweepError as exc:
        raise SweepError(str(exc)) from exc
    return len({CellKey.of(row).model_dump_json() for row in rows})


def _pool_digest(pool: ModelPool) -> str:
    """A digest of the candidate roster: which models, at which prices, in which order.

    The whole entry, not just the name: a matrix's rows are priced by their pool entry, so an
    edited price makes an old row's `cost_usd` a different number than the same cell would get
    today, and resuming across that edit would mix two price regimes into one comparison.

    `enabled` is excluded from the digest: the pool digested here is already filtered to the
    enabled entries, so the field carries no information, and hashing it would break every
    pre-0.2.3 partial sidecar for an otherwise UNCHANGED roster the day the field was added
    (a serialization-shape change, not a roster change). Turning an entry off still re-plans,
    because the entry leaves the filtered pool entirely.
    """
    payload = "\n".join(entry.model_dump_json(exclude={"enabled"}) for entry in pool.models)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _persist_harvest(harvest: _SessionHarvest, runs_dir: Path) -> tuple[RunRecord, Path | None]:
    """Roll the harvested sessions into one record and write it, unless nothing was metered."""
    usage = merge_run_records(
        harvest.records,
        run_id=f"sweep-{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}",
        kind=SWEEP_RUN_KIND,
    )
    return usage, (save_run(usage, runs_dir) if harvest.records else None)


SWEEP_RUN_KIND = "sweep"
"""`RunRecord.kind` for a sweep's world-model side, beside the existing "build" and "serve"."""


class _SessionHarvest:
    """An `env_factory` wrapper that collects each episode's world-model usage as it completes.

    Harvest happens at each env's OWN close, which is the only rule that survives concurrency.
    The previous design banked the last env's record when the next env was requested, on the
    assumption that exactly one env is alive at a time; with cells running in parallel that
    assumption is false in both directions (several envs are open at once, and the next request
    comes from a different thread than the one that finished), and the records would be attributed
    to the wrong episodes or dropped entirely. Wrapping the env instead needs no bookkeeping at
    all: the record is final exactly when `close` returns, and that is where it is taken.

    An env that exposes no usage is counted, not guessed at. A cell accepts any `Env`, and a
    caller who supplies a non-metering one gets a stated gap rather than a fabricated $0.
    """

    def __init__(self, inner: Callable[[], Env]) -> None:
        self._inner = inner
        self._lock = threading.Lock()
        self.records: list[RunRecord] = []
        self.unmetered = 0

    def __call__(self) -> Env:
        """Build one cell's env, wrapped so its record is banked when the cell closes it."""
        return cast("Env", _HarvestedEnv(self._inner(), self))

    def bank(self, env: Env) -> None:
        """Take the final record off a closed env, or count it as unmetered."""
        record = getattr(env, "usage", None)
        with self._lock:
            if isinstance(record, RunRecord):
                self.records.append(record)
            else:
                self.unmetered += 1


class _HarvestedEnv:
    """One cell's env, which banks its world-model usage into the harvest when it closes.

    Everything delegates. `last_score` in particular must propagate its own failures verbatim,
    since a cell reads it defensively and tells a missing attribute (a non-scoring env, fatal)
    apart from a raised one (a throttled judge, this cell only).
    """

    def __init__(self, inner: Env, harvest: _SessionHarvest) -> None:
        self._inner = inner
        self._harvest = harvest
        self._banked = False

    def reset(self, task: str | None = None, seed_state: EnvState | None = None) -> EnvState:
        return self._inner.reset(task=task, seed_state=seed_state)

    def step(self, action: Action) -> Observation:
        return self._inner.step(action)

    def close(self) -> None:
        """Close the episode, then bank its record. Idempotent, as the `Env` contract requires."""
        try:
            self._inner.close()
        finally:
            if not self._banked:
                self._banked = True
                self._harvest.bank(self._inner)

    @property
    def last_score(self) -> EpisodeScore:
        scoring = cast("ScoringEnv", self._inner)
        return scoring.last_score

    @property
    def usage(self) -> RunRecord | None:
        record = getattr(self._inner, "usage", None)
        return record if isinstance(record, RunRecord) else None


class CandidateCoverage(BaseModel):
    """One candidate's scored coverage: what a fitter would actually weigh it on."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate: str
    scored: int  # cells with a verified reward
    unscored: int  # cells whose episode or scoring failed (skipped by both fitters)
    # Scored EPISODE COUNT per swept scenario, in sweep order. Counts, not presence: both fitters
    # weigh episodes, not scenarios (see `unevenness`), so `X: 3` and `X: 1` are different
    # evidence even though both "cover" X.
    scored_episodes: tuple[tuple[str, int], ...]
    first_error: str | None  # first error text among this candidate's unscored cells, if any

    @property
    def lost_scenarios(self) -> tuple[str, ...]:
        """Swept scenarios this candidate has NO scored episode for."""
        return tuple(sid for sid, count in self.scored_episodes if count == 0)


class Unevenness(StrEnum):
    """How the candidates' scored evidence differs, when it does."""

    EVEN = "even"
    SCENARIOS = "scenarios"  # candidates were scored on different scenario SETS
    EPISODES = "episodes"  # same scenarios, different numbers of scored episodes


def coverage(matrix: OutcomeMatrix) -> list[CandidateCoverage]:
    """Per-candidate scored coverage over the swept scenarios, in pool order."""
    swept = matrix.scenario_ids()
    rows: list[CandidateCoverage] = []
    for name in matrix.model_names():
        cells = [outcome for outcome in matrix.outcomes if outcome.model == name]
        per_scenario: Counter[str] = Counter(cell.scenario_id for cell in cells if cell.scored)
        errors = [cell.error for cell in cells if not cell.scored and cell.error]
        rows.append(
            CandidateCoverage(
                candidate=name,
                scored=sum(1 for cell in cells if cell.scored),
                unscored=sum(1 for cell in cells if not cell.scored),
                scored_episodes=tuple((sid, per_scenario[sid]) for sid in swept),
                first_error=errors[0] if errors else None,
            )
        )
    return rows


def unevenness(rows: list[CandidateCoverage]) -> Unevenness:
    """Whether the candidates were scored on the same evidence, and if not, how it differs.

    Compared as per-(candidate, scenario) scored EPISODE COUNTS, because that is what the fitters
    weigh. Presence is not enough: `fit_rank_policy` averages every surviving EPISODE
    independently, so a candidate that kept 3 episodes on a scenario and one that kept 1 carry
    different effective scenario weights into the same cluster mean, and the models chosen as
    `default_model` (`routing.py:_overall_best`) and as the knn fallback/guard
    (`knn.py:best_single_on_fit`) are picked off those same episode-weighted means. The knn BANK is
    milder, since its cells are per-scenario means, but "milder" is not "unbiased", and it is the
    same matrix either way.

    Losing the same episodes of the same scenarios for EVERY candidate is even: the comparison
    stays like-for-like on less data, and the counts show the loss.
    """
    if len({row.scored_episodes for row in rows}) <= 1:
        return Unevenness.EVEN
    if len({row.lost_scenarios for row in rows}) > 1:
        return Unevenness.SCENARIOS
    return Unevenness.EPISODES


def _check_out_writable(out_path: Path) -> None:
    """Prove the matrix destination can be written BEFORE the sweep spends anything.

    `OutcomeMatrix.save` creates the parent directory and writes, but only once every episode has
    run: a destination that cannot be created (a parent component that is a regular file, an
    unwritable directory, a path that is itself a directory) would throw the whole paid sweep
    away with an OS error and no message. Checked without creating anything, so declining the
    cost confirmation still leaves the filesystem untouched.

    Raises:
        SweepError: The matrix could not be written there.
    """
    resolved = out_path if out_path.is_absolute() else Path.cwd() / out_path
    existing = resolved.parent
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    problem: str | None = None
    if not existing.is_dir():
        problem = f"{existing} is not a directory"
    elif not os.access(existing, os.W_OK):
        problem = f"{existing} is not writable"
    elif resolved.is_dir():
        problem = f"{resolved} is a directory"
    elif resolved.exists() and not os.access(resolved, os.W_OK):
        problem = f"{resolved} is not writable"
    if problem is not None:
        raise SweepError(
            f"cannot write the outcome matrix to {out_path}: {problem}. The matrix destination "
            "must be a writable JSON file path"
        )


def _corpus_traces(model_dir: Path, adapter_name: str, explicit: Path | None) -> list[Trace]:
    """Ingest the corpus the sweep takes its scenarios from: the explicit file, else the model's.

    A build does NOT persist the corpus it read (it keeps prompts, metrics and the retrieval
    index), so `local_traces_path` finds a file only for a Hub-downloaded model or a shipped
    example. `--traces` is the same escape hatch `wmo demo` carries for exactly this reason, and
    the failure names it.
    """
    if explicit is not None:
        if not explicit.is_file():
            raise SweepError(f"no trace file at {explicit} (--traces)")
        path = explicit
    else:
        found = local_traces_path(model_dir)
        if found is None:
            raise SweepError(
                f"no trace corpus for world model '{model_dir.name}': looked for "
                f"{model_dir / TRACES_FILENAME} and {model_dir.parent.parent / TRACES_FILENAME}. "
                "Sweep scenarios are the model's OWN held-out task prompts, and a build keeps no "
                "copy of the corpus it read, so pass `--traces <the file wmo build --file read>` "
                f"or put that file at {model_dir / TRACES_FILENAME}"
            )
        path = found
    try:
        traces = get_adapter(adapter_name).from_file(str(path))
    except (OSError, ValueError) as exc:  # unknown adapter, unreadable or malformed corpus
        raise SweepError(f"cannot ingest {path}: {exc}") from exc
    if not traces:
        raise SweepError(
            f"{path} ingested no traces with the '{adapter_name}' adapter, so there are no "
            "scenarios to sweep"
        )
    return traces


def _estimate_cost(
    pool: ModelPool,
    *,
    episodes_per_candidate: int,
    calls_per_episode: int,
    input_tokens: int,
    output_tokens: int,
) -> list[CostLine]:
    """Project each candidate's spend, priced by its OWN pool entry (overrides included)."""
    per_call = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
    calls = episodes_per_candidate * calls_per_episode
    lines: list[CostLine] = []
    for entry in pool.models:
        price = entry.price()
        lines.append(
            CostLine(
                candidate=entry.name,
                episodes=episodes_per_candidate,
                calls=calls,
                input_per_mtok=price.input_per_mtok,
                output_per_mtok=price.output_per_mtok,
                usd=calls * entry.call_cost_usd(per_call),
            )
        )
    return lines
