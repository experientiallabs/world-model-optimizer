"""Closed-loop evaluation: roll every pool candidate over a scenario set, collect the matrix.

Each (candidate model, scenario, episode) cell runs one `run_episode` with the candidate driving
`LLMAgent` against the env, then reads the env's episode score (VERIFY). The result is the
`OutcomeMatrix` the routing optimizer fits on and the improvement report cites.

The grid is addressable, not just iterable: `pool_cells` enumerates it in the canonical order,
`run_cell` measures exactly one cell, and `run_cells` drives a list of them with bounded
concurrency. That split is what lets `wmo.optimize.sweep` resume a partially measured grid (it
runs the cells that are missing, not the whole product) while `evaluate_pool` stays the one-call
API for measuring all of it.

Concurrency notes:
- A cell is an independent episode against its OWN env and its OWN candidate provider, so cells
  parallelize; `max_concurrency` bounds how many are in flight. The default of 1 is the
  sequential loop this module has always run, cell for cell.
- The work is IO-bound (provider round trips), so the pool is threads, not processes.
- Completion order is nondeterministic above concurrency 1, but the returned outcomes are always
  in the canonical cell order, so the matrix (and every digest taken of it) does not depend on
  which cell finished first.
- `on_outcome` fires from the WORKER thread that finished the cell, serialized under one lock, so
  a console-printing callback never interleaves two cells mid-line.

Measurement notes:
- Latency is measured per POLICY CALL (the candidate's own completions), not per episode: episode
  wall time is dominated by the world model's simulation latency, which production traffic never
  pays, so quoting it would flatter nobody honestly.
- Cost is the candidate side only, priced by its own pool entry; the env's serve/judge cost is
  metered separately by the world model (D12 cost split).
- Every raw candidate reply is stored (`ScenarioOutcome.replies`): that is the future
  distillation feed. Providers do not yet surface separated thinking blocks; when they do, the
  capture point is `_TimedProvider.complete`.
- A cell is recorded SCORED only when the episode also ran clean. An episode that errored
  mid-flight (provider throttle, agent crash) is unscored even if the env still produced a
  score, because that score grades a run the candidate never got to finish: counting it as
  reward would read infrastructure failure as incapability. The salvaged critique is kept on
  the row, prefixed, so the diagnostic text is not lost.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Protocol, cast

from pydantic import BaseModel, ConfigDict

from wmo.core.types import ActionKind
from wmo.env.base import Env
from wmo.env.episode import run_episode
from wmo.env.llm_agent import DEFAULT_HISTORY_CHARS, LLMAgent
from wmo.env.scenarios import Scenario
from wmo.optimize.compression import CompressionConfig, estimate_tokens, get_compressor
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.reward import EpisodeScore
from wmo.providers.base import (
    DEFAULT_MAX_TOKENS,
    Completion,
    Message,
    Provider,
    ProviderConfig,
    TokenUsage,
    VerifyResult,
)
from wmo.providers.pool import ModelPool, PoolEntry, pool_provider

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)


class _TimedProvider:
    """Wraps the candidate's provider to record per-call seconds, usage, and raw replies."""

    def __init__(self, provider: Provider) -> None:
        self._provider = provider
        self.call_seconds: list[float] = []
        self.call_input_tokens: list[int] = []
        self.call_output_tokens: list[int] = []
        self.call_cached_input_tokens: list[int] = []
        self.call_cache_write_input_tokens: list[int] = []
        self.call_usage: list[TokenUsage] = []
        self.replies: list[str] = []
        self.usage = TokenUsage()

    @property
    def config(self) -> ProviderConfig:
        return self._provider.config

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Completion:
        started = time.monotonic()
        completion = self._provider.complete(
            system, messages, temperature=temperature, max_tokens=max_tokens
        )
        self.call_seconds.append(time.monotonic() - started)
        self.call_input_tokens.append(completion.usage.input_tokens)
        self.call_output_tokens.append(completion.usage.output_tokens)
        self.call_cached_input_tokens.append(completion.usage.cached_input_tokens)
        self.call_cache_write_input_tokens.append(completion.usage.cache_write_input_tokens)
        self.call_usage.append(completion.usage)
        self.replies.append(completion.text)
        self.usage = TokenUsage(
            input_tokens=self.usage.input_tokens + completion.usage.input_tokens,
            cached_input_tokens=self.usage.cached_input_tokens
            + completion.usage.cached_input_tokens,
            cache_write_input_tokens=self.usage.cache_write_input_tokens
            + completion.usage.cache_write_input_tokens,
            output_tokens=self.usage.output_tokens + completion.usage.output_tokens,
            reasoning_tokens=self.usage.reasoning_tokens + completion.usage.reasoning_tokens,
        )
        return completion

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._provider.embed(texts)

    def verify(self) -> VerifyResult:
        return self._provider.verify()


class _CompressingProvider:
    """Applies the D-COMPRESS stage to the candidate's calls, so eval measures through it.

    Sits ABOVE `_TimedProvider` (agent -> compressing -> timed -> real): the timed layer then
    records the latency and provider-reported usage of the ACTUAL, compressed call. Mirrors
    the serving rule exactly: only user-role message content is compressed; the system prompt
    and the model's own prior replies pass through verbatim. Determinism keeps the growing
    transcript append-stable across the episode's calls, exactly as serving's affinity-miss
    path reproduces bytes. Accumulates the compressor's own accounting per episode.
    """

    def __init__(self, provider: Provider, config: CompressionConfig) -> None:
        self._provider = provider
        self._config = config
        self.compressor = get_compressor(config.compressor_id)
        self.tokens_in_raw = 0
        self.tokens_in_compressed = 0
        self.latency_s = 0.0
        self.cost_usd = 0.0

    @property
    def config(self) -> ProviderConfig:
        return self._provider.config

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Completion:
        started = time.monotonic()
        user_segments = [m.content for m in messages if m.role == "user"]
        result = self.compressor.compress(user_segments, self._config)
        replacements = iter(result.segments)
        compressed = [
            Message(role="user", content=next(replacements)) if m.role == "user" else m
            for m in messages
        ]
        self.tokens_in_raw += sum(estimate_tokens(m.content) for m in messages)
        self.tokens_in_compressed += sum(estimate_tokens(m.content) for m in compressed)
        self.cost_usd += result.cost_usd
        self.latency_s += time.monotonic() - started
        return self._provider.complete(
            system, compressed, temperature=temperature, max_tokens=max_tokens
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._provider.embed(texts)

    def verify(self) -> VerifyResult:
        return self._provider.verify()


_NEEDS_SCORING_ENV = (
    "env produced no episode score; evaluate_pool needs a scoring env "
    "(e.g. WorldModelEnv(world_model, score_on_close=True))"
)

# Prefix on a critique salvaged from an episode that errored: the text is diagnostic, the
# verdict is not evidence (see the module docstring).
_SALVAGE_PREFIX = "salvage-judged despite error: "


class ScoringEnv(Protocol):
    """The scoring surface a cell needs on top of `Env`; `WorldModelEnv` provides it."""

    @property
    def last_score(self) -> EpisodeScore: ...


def _read_episode_score(env: Env) -> tuple[EpisodeScore | None, str | None]:
    """Read `env.last_score` defensively, returning (score, reason it is missing).

    The two failure modes are NOT the same failure. A missing attribute means the caller handed
    `evaluate_pool` a non-scoring env, so no cell of the sweep can ever be evidence and the
    error is fatal. A `last_score` that RAISES (WorldModelEnv re-raises a close-time scoring
    failure, e.g. a throttled judge call) is per-episode: that one cell is unscored with the
    reason recorded, and the sweep keeps its other completed cells.
    """
    scoring = cast("ScoringEnv", env)
    try:
        raw: object = scoring.last_score
    except AttributeError as exc:
        raise ValueError(_NEEDS_SCORING_ENV) from exc
    except RuntimeError as exc:
        return None, f"episode scoring failed: {exc}"
    if raw is None:
        return None, None
    if not isinstance(raw, EpisodeScore):
        # An unscored row must always say WHY (the outcomes contract); a wrong-typed score
        # silently becoming reward=None/error=None would violate it.
        return None, (
            f"env last_score is {type(raw).__name__}, not EpisodeScore; episode left unscored"
        )
    return raw, None


def scenario_id(scenario: Scenario) -> str:
    """Stable id for a scenario: its first provenance trace id, else a hash of the task.

    Provisional until wm-create's generate contract ships first-class scenario ids
    (DECISIONS.md 2026-07-23); both forms are deterministic across runs.
    """
    if scenario.provenance:
        return scenario.provenance[0]
    return hashlib.sha256(scenario.task.encode("utf-8")).hexdigest()[:12]


class CellKey(BaseModel):
    """Which cell a row of the matrix is: one scenario, one candidate, one episode index.

    The identity a partially measured grid is resumed against, so it is a value rather than a
    tuple: `(scenario, model, episode)` in either order reads the same in a type signature, and
    reading it wrong is how a resume silently re-buys or silently skips the wrong cells.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str
    model: str  # pool entry name
    episode: int

    @classmethod
    def of(cls, outcome: ScenarioOutcome) -> CellKey:
        """The key of a measured row."""
        return cls(scenario_id=outcome.scenario_id, model=outcome.model, episode=outcome.episode)

    def __str__(self) -> str:
        return f"{self.model} on {self.scenario_id} ep{self.episode}"


class PoolCell(BaseModel):
    """One unit of closed-loop work: this candidate, on this scenario, for this episode index."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    entry: PoolEntry
    scenario: Scenario
    episode: int
    # Set when this cell is being measured AGAIN after an earlier attempt (a transport fault, a
    # throttled judge). Rides onto the row so a matrix says which of its cells are re-runs: the
    # retry of a flaky cell and a cell that ran clean the first time are not the same evidence.
    remeasured: bool = False

    @property
    def key(self) -> CellKey:
        return CellKey(
            scenario_id=scenario_id(self.scenario), model=self.entry.name, episode=self.episode
        )


def pool_cells(
    pool: ModelPool, scenarios: list[Scenario], *, episodes_per_scenario: int = 1
) -> list[PoolCell]:
    """Every cell of the grid, in the canonical order: pool order, scenario order, episode index.

    THE order, not an order: it is what a matrix's rows are sorted into whatever sequence they
    were measured in, so two runs of the same plan at different concurrencies produce the same
    file. Callers that measure a subset (a resume, a retry pass) filter this list rather than
    building their own, so the subset's rows still land in the same places.
    """
    return [
        PoolCell(entry=entry, scenario=scenario, episode=episode)
        for entry in pool.models
        for scenario in scenarios
        for episode in range(episodes_per_scenario)
    ]


def run_cell(
    cell: PoolCell,
    env_factory: Callable[[], Env],
    *,
    max_steps: int = 20,
    agent_temperature: float = 0.0,
    tools_hint: str | None = None,
    history_chars: int = DEFAULT_HISTORY_CHARS,
    provider_factory: Callable[[PoolEntry], Provider] = pool_provider,
    compression: CompressionConfig | None = None,
) -> ScenarioOutcome:
    """Measure ONE cell: one episode of one candidate on one scenario, against a fresh env.

    Everything the cell owns is built here rather than shared: its candidate provider (so
    per-episode provider state stays per episode, and so concurrent cells never share a client),
    its agent, and its env. That is what makes a cell safe to run on a worker thread.

    Raises:
        ValueError: The env produced no episode score and no error, so it does not score at all
            and no cell of this measurement could ever be evidence.
    """
    timed = _TimedProvider(provider_factory(cell.entry))
    compressing = _CompressingProvider(timed, compression) if compression is not None else None
    agent = LLMAgent(
        compressing if compressing is not None else timed,
        temperature=agent_temperature,
        tools_hint=tools_hint,
        history_chars=history_chars,
    )
    env = env_factory()
    result = run_episode(env, agent, cell.scenario.task, max_steps=max_steps)
    score, score_error = _read_episode_score(env)
    error = result.error
    if score_error is not None:
        error = f"{error}; {score_error}" if error else score_error
    if score is None and error is None:
        raise ValueError(_NEEDS_SCORING_ENV)
    critique = score.critique if score else ""
    if score is not None and result.error is not None:
        # The episode broke mid-flight, so this verdict grades an unfinished run: keep the text,
        # drop the reward (see the module docstring).
        critique = f"{_SALVAGE_PREFIX}{critique}" if critique else ""
        score = None
    return ScenarioOutcome(
        scenario_id=scenario_id(cell.scenario),
        task=cell.scenario.task,
        model=cell.entry.name,
        episode=cell.episode,
        reward=score.reward if score else None,
        success=score.success if score else False,
        critique=critique,
        steps=len(result.steps),
        tool_calls=sum(step.action.kind is ActionKind.TOOL_CALL for step in result.steps),
        stop_reason=str(result.stop_reason),
        usage=timed.usage,
        cost_usd=sum(cell.entry.call_cost_usd(usage) for usage in timed.call_usage),
        call_seconds=timed.call_seconds,
        call_input_tokens=timed.call_input_tokens,
        call_output_tokens=timed.call_output_tokens,
        call_cached_input_tokens=timed.call_cached_input_tokens,
        call_cache_write_input_tokens=timed.call_cache_write_input_tokens,
        replies=timed.replies,
        error=error,
        remeasured=cell.remeasured,
        tokens_in_raw=compressing.tokens_in_raw if compressing else 0,
        tokens_in_compressed=compressing.tokens_in_compressed if compressing else 0,
        compressor_id=compressing.compressor.id if compressing else "",
        compressor_version=compressing.compressor.version if compressing else "",
        aggressiveness=compression.aggressiveness if compression else 0.0,
        compressor_latency_s=compressing.latency_s if compressing else 0.0,
        compressor_cost_usd=compressing.cost_usd if compressing else 0.0,
    )


def run_cells(
    cells: Sequence[PoolCell],
    env_factory: Callable[[], Env],
    *,
    max_steps: int = 20,
    agent_temperature: float = 0.0,
    tools_hint: str | None = None,
    history_chars: int = DEFAULT_HISTORY_CHARS,
    provider_factory: Callable[[PoolEntry], Provider] = pool_provider,
    on_outcome: Callable[[ScenarioOutcome], None] | None = None,
    compression: CompressionConfig | None = None,
    max_concurrency: int = 1,
) -> list[ScenarioOutcome]:
    """Measure `cells`, at most `max_concurrency` at a time, and return them IN CELL ORDER.

    `max_concurrency=1` is the plain sequential loop, with no thread and no executor involved:
    the default path is the one this module has always run. Above 1 the cells go to a thread pool
    (the work is provider round trips, so threads are the right tool) and the returned list is
    still indexed by input position, never by finish time.

    When a cell raises, the cells already in flight are allowed to finish (they are being paid
    for either way) and no queued cell is started; the first failure is then re-raised. A caller
    that wants the completed rows despite the failure collects them from `on_outcome`.

    `on_outcome` fires as each cell completes, from the worker thread that ran it, serialized
    under one lock so two cells cannot interleave inside it. A callback that raises is logged and
    ignored: a broken progress pipe (or a full disk under a persisting callback) must not throw
    away cells that have already been paid for.

    Raises:
        ValueError: `max_concurrency` is below 1, or a cell's env does not score.
    """
    if max_concurrency < 1:
        raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
    measured: dict[int, ScenarioOutcome] = {}
    lock = threading.Lock()

    def measure(index: int, cell: PoolCell) -> None:
        """Run one cell and bank it, with the reporting every path shares."""
        outcome = run_cell(
            cell,
            env_factory,
            max_steps=max_steps,
            agent_temperature=agent_temperature,
            tools_hint=tools_hint,
            history_chars=history_chars,
            provider_factory=provider_factory,
            compression=compression,
        )
        with lock:
            measured[index] = outcome
            if on_outcome is not None:
                try:
                    on_outcome(outcome)
                except Exception:  # noqa: BLE001 - a broken progress pipe never costs cells
                    logger.warning(
                        "on_outcome callback failed for %s; measurement continues",
                        cell.key,
                        exc_info=True,
                    )
            logger.info(
                "closed-loop %s: reward=%s cost=$%.5f steps=%d",
                cell.key,
                "unscored" if outcome.reward is None else f"{outcome.reward:.2f}",
                outcome.cost_usd,
                outcome.steps,
            )

    if max_concurrency == 1:
        for index, cell in enumerate(cells):
            measure(index, cell)
    else:
        _measure_concurrently(cells, measure, max_concurrency=max_concurrency)
    return [measured[index] for index in sorted(measured)]


def _measure_concurrently(
    cells: Sequence[PoolCell],
    measure: Callable[[int, PoolCell], None],
    *,
    max_concurrency: int,
) -> None:
    """Drive `measure` over `cells` on a bounded thread pool, re-raising the first failure.

    Cancelling the queue rather than the pool is the point: a cell already in flight has an open
    world-model session and a candidate call in the air, so killing it would spend money and
    record nothing. Queued cells have spent nothing, so they are dropped.
    """
    failure: BaseException | None = None
    with ThreadPoolExecutor(max_workers=max_concurrency, thread_name_prefix="wmo-cell") as pool:
        futures: dict[Future[None], PoolCell] = {
            pool.submit(measure, index, cell): cell for index, cell in enumerate(cells)
        }
        for future in as_completed(futures):
            try:
                future.result()
            except BaseException as exc:  # noqa: BLE001 - re-raised below, once the pool drains
                if failure is None:
                    failure = exc
                    for queued in futures:
                        queued.cancel()
    if failure is not None:
        raise failure


def evaluate_pool(
    env_factory: Callable[[], Env],
    pool: ModelPool,
    scenarios: list[Scenario],
    *,
    episodes_per_scenario: int = 1,
    max_steps: int = 20,
    agent_temperature: float = 0.0,
    tools_hint: str | None = None,
    history_chars: int = DEFAULT_HISTORY_CHARS,
    provider_factory: Callable[[PoolEntry], Provider] = pool_provider,
    on_outcome: Callable[[ScenarioOutcome], None] | None = None,
    compression: CompressionConfig | None = None,
    max_concurrency: int = 1,
) -> OutcomeMatrix:
    """Run every pool candidate over `scenarios`, one fresh env per episode.

    The env must score episodes on close (`WorldModelEnv(..., score_on_close=True)`): a matrix
    without verified rewards is not evidence. Episodes that error, and episodes whose scoring
    itself fails, are recorded unscored (`reward=None`, `error` set) rather than defaulted to 0,
    and never abort the sweep. `on_outcome` fires after each cell for progress display; a
    callback that raises is logged and ignored, since a broken progress pipe must not throw away
    the cells already paid for.

    `history_chars` is how much of each observation the agent gets to see before the next turn
    (`LLMAgent`). It is exposed here because it is environment-dependent, not a constant: the
    default was raised to 2000 after 500 truncated tau-bench payloads into verbatim re-fetch
    loops, and a corpus with larger observations needs more still. It changes what the candidates
    are measured on, so it is part of a matrix's capture cohort.

    `compression` applies the D-COMPRESS stage to every candidate call, measured through the
    same wrapper stack production serves through; each outcome then carries the compressor's
    per-episode accounting fields. None (the default) = uncompressed, today's rows exactly.

    `max_concurrency` bounds how many cells are in flight at once (1 = sequential, the default).
    It changes only how long the measurement takes, never what a cell measures or where its row
    lands, so two matrices that differ only in this value are the same evidence. The real ceiling
    is your provider's rate limit, and the world model's own serve and judge calls all come out of
    ONE account's bucket.
    """
    return OutcomeMatrix(
        pool=pool.models,
        outcomes=run_cells(
            pool_cells(pool, scenarios, episodes_per_scenario=episodes_per_scenario),
            env_factory,
            max_steps=max_steps,
            agent_temperature=agent_temperature,
            tools_hint=tools_hint,
            history_chars=history_chars,
            provider_factory=provider_factory,
            on_outcome=on_outcome,
            compression=compression,
            max_concurrency=max_concurrency,
        ),
    )
