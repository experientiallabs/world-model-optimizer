"""Closed-loop evaluation: roll every pool candidate over a scenario set, collect the matrix.

Each (candidate model, scenario, episode) cell runs one `run_episode` with the candidate driving
`LLMAgent` against the env, then reads the env's episode score (VERIFY). The result is the
`OutcomeMatrix` the routing optimizer fits on and the improvement report cites.

Measurement notes:
- Cells run candidate-MINOR (scenario, then episode, then candidate), so every candidate runs its
  Nth cell before any candidate runs its (N+1)th. The order is ours to choose only because the
  caller freezes the world model (see `evaluate_pool`), and this is the choice that bounds what a
  backend which can only fail once it is CALLED costs before anyone sees it fail.
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
import time
from typing import TYPE_CHECKING, Protocol, cast

from wmo.env.base import Env
from wmo.env.episode import run_episode
from wmo.env.llm_agent import LLMAgent
from wmo.env.scenarios import Scenario
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
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class _TimedProvider:
    """Wraps the candidate's provider to record per-call seconds, usage, and raw replies."""

    def __init__(self, provider: Provider) -> None:
        self._provider = provider
        self.call_seconds: list[float] = []
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
        self.replies.append(completion.text)
        self.usage = TokenUsage(
            input_tokens=self.usage.input_tokens + completion.usage.input_tokens,
            cached_input_tokens=self.usage.cached_input_tokens
            + completion.usage.cached_input_tokens,
            output_tokens=self.usage.output_tokens + completion.usage.output_tokens,
        )
        return completion

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


class _ScoringEnv(Protocol):
    """The scoring surface `evaluate_pool` needs on top of `Env`; `WorldModelEnv` provides it."""

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
    scoring = cast("_ScoringEnv", env)
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


def evaluate_pool(
    env_factory: Callable[[], Env],
    pool: ModelPool,
    scenarios: list[Scenario],
    *,
    episodes_per_scenario: int = 1,
    max_steps: int = 20,
    agent_temperature: float = 0.0,
    tools_hint: str | None = None,
    provider_factory: Callable[[PoolEntry], Provider] = pool_provider,
    on_outcome: Callable[[ScenarioOutcome], None] | None = None,
) -> OutcomeMatrix:
    """Run every pool candidate over `scenarios`, one fresh env per episode.

    The env must score episodes on close (`WorldModelEnv(..., score_on_close=True)`): a matrix
    without verified rewards is not evidence. Episodes that error, and episodes whose scoring
    itself fails, are recorded unscored (`reward=None`, `error` set) rather than defaulted to 0,
    and never abort the sweep. `on_outcome` fires after each cell for progress display; a
    callback that raises is logged and ignored, since a broken progress pipe must not throw away
    the cells already paid for.

    Cells run candidate-MINOR: scenario, then episode, then candidate. Every candidate therefore
    runs its Nth cell before any candidate runs its (N+1)th, which is what bounds a backend that
    can only fail once it is CALLED. Bedrock with no AWS credentials is the measured example
    (boto3 walks a credential chain that reaches the instance-metadata endpoint over the network
    and builds a client with no credentials at all, so no request-free pre-flight can see it):
    that candidate now surfaces through `on_outcome` after one cell per earlier candidate,
    instead of after every earlier candidate has run every one of its cells. Its own remaining
    cells still run, because one throttled call is not a dead backend and the paid cells are the
    diagnosis either way; a caller that wants to stop watches `on_outcome`.

    ORDER SAFETY: which order to run cells in is only ours to choose while the environment cannot
    LEARN between them. `wmo optimize route sweep` wraps this whole call in
    `world_model.frozen()`, which suspends index enrichment and knowledge writes, so no cell's
    predicted steps become another cell's retrieved demos and the matrix is the same whatever
    order produced it. Against an enriching env the scores are order-dependent either way, and
    this order changes which cells feed which: freeze the model, or open its sessions with
    `enrich=False`.
    """
    outcomes: list[ScenarioOutcome] = []
    for scenario in scenarios:
        sid = scenario_id(scenario)
        for episode in range(episodes_per_scenario):
            for entry in pool.models:
                timed = _TimedProvider(provider_factory(entry))
                agent = LLMAgent(timed, temperature=agent_temperature, tools_hint=tools_hint)
                env = env_factory()
                result = run_episode(env, agent, scenario.task, max_steps=max_steps)
                score, score_error = _read_episode_score(env)
                error = result.error
                if score_error is not None:
                    error = f"{error}; {score_error}" if error else score_error
                if score is None and error is None:
                    raise ValueError(_NEEDS_SCORING_ENV)
                critique = score.critique if score else ""
                if score is not None and result.error is not None:
                    # The episode broke mid-flight, so this verdict grades an unfinished run:
                    # keep the text, drop the reward (see the module docstring).
                    critique = f"{_SALVAGE_PREFIX}{critique}" if critique else ""
                    score = None
                outcome = ScenarioOutcome(
                    scenario_id=sid,
                    task=scenario.task,
                    model=entry.name,
                    episode=episode,
                    reward=score.reward if score else None,
                    success=score.success if score else False,
                    critique=critique,
                    steps=len(result.steps),
                    stop_reason=str(result.stop_reason),
                    usage=timed.usage,
                    cost_usd=entry.cost_usd(timed.usage),
                    call_seconds=timed.call_seconds,
                    replies=timed.replies,
                    error=error,
                )
                outcomes.append(outcome)
                if on_outcome is not None:
                    try:
                        on_outcome(outcome)
                    except Exception:  # noqa: BLE001 - a broken progress pipe never costs cells
                        logger.warning(
                            "on_outcome callback failed for %s on %s ep%d; sweep continues",
                            entry.name,
                            sid,
                            episode,
                            exc_info=True,
                        )
                logger.info(
                    "closed-loop %s on %s ep%d: reward=%s cost=$%.5f steps=%d",
                    entry.name,
                    sid,
                    episode,
                    "unscored" if outcome.reward is None else f"{outcome.reward:.2f}",
                    outcome.cost_usd,
                    outcome.steps,
                )
    return OutcomeMatrix(pool=pool.models, outcomes=outcomes)
