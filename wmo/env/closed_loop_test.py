"""Tests for the closed-loop pool evaluation (candidate models rolled against an Env)."""

from __future__ import annotations

from typing import cast

import pytest

from wmo.core.types import Action, EnvState, Observation
from wmo.env.closed_loop import evaluate_pool
from wmo.env.scenarios import Scenario
from wmo.optimize.outcomes import ScenarioOutcome
from wmo.optimize.reward import EpisodeScore
from wmo.providers.base import (
    Completion,
    Message,
    ProviderKind,
    TokenUsage,
    VerifyResult,
)
from wmo.providers.pool import ModelPool, PoolEntry


class _FakeEnv:
    """Scripted Env: every action gets one observation; scoring is canned."""

    def __init__(self, score: EpisodeScore | None) -> None:
        self.last_score = score

    def reset(self, task: str | None = None, seed_state: EnvState | None = None) -> EnvState:
        return EnvState()

    def step(self, action: Action) -> Observation:
        return Observation(content="ok")

    def close(self) -> None:
        return None


class _ThrottledScoringEnv:
    """Env shaped like WorldModelEnv: `last_score` is a property that RAISES when scoring failed.

    A throttled judge call at close time surfaces exactly this way, and it must cost one cell,
    not the sweep.
    """

    @property
    def last_score(self) -> EpisodeScore:
        raise RuntimeError("Bedrock ThrottlingException while scoring the session")

    def reset(self, task: str | None = None, seed_state: EnvState | None = None) -> EnvState:
        return EnvState()

    def step(self, action: Action) -> Observation:
        return Observation(content="ok")

    def close(self) -> None:
        return None


class _ScriptedProvider:
    """Provider whose completions are a fixed script (one tool call, then done)."""

    def __init__(self, entry: PoolEntry) -> None:
        self.config = entry.provider_config()
        self._script = [
            '{"tool": "ls", "arguments": {}}',
            '{"done": true, "summary": "finished"}',
        ]
        self._i = 0

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        text = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        return Completion(text=text, usage=TokenUsage(input_tokens=10, output_tokens=5))

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def verify(self) -> VerifyResult:
        raise NotImplementedError


class _FailFirstProvider(_ScriptedProvider):
    """Provider that throws on its first completion: an agent-side failure mid-episode."""

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        raise RuntimeError("RateLimitError: 429 too many requests")


class _UnusableBackendProvider(_ScriptedProvider):
    """Provider that raises the moment it is CALLED: the missing-AWS-credentials shape.

    The one failure a request-free pre-flight cannot catch. boto3 resolves credentials by walking
    a chain that reaches the instance-metadata endpoint over the network, and builds a Bedrock
    client with no credentials at all, so the entry looks fine until a cell is spent on it.
    """

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        raise RuntimeError("NoCredentialsError: Unable to locate credentials")


def _pool() -> ModelPool:
    return ModelPool(
        models=[
            PoolEntry(
                name="candidate-a",
                kind=ProviderKind.OPENAI,
                model="custom-a",
                input_per_mtok=1.0,
                output_per_mtok=2.0,
            ),
            PoolEntry(
                name="candidate-b",
                kind=ProviderKind.OPENAI,
                model="custom-b",
                input_per_mtok=10.0,
                output_per_mtok=20.0,
            ),
        ]
    )


_SCENARIOS = [
    Scenario(task="list the files", provenance=["trace-1"]),
    Scenario(task="delete the files", provenance=["trace-2"]),
]


def test_evaluate_pool_builds_full_matrix() -> None:
    matrix = evaluate_pool(
        lambda: _FakeEnv(EpisodeScore(reward=0.8, success=True, critique="fine")),
        _pool(),
        _SCENARIOS,
        provider_factory=_ScriptedProvider,
        max_steps=5,
    )

    assert matrix.model_names() == ["candidate-a", "candidate-b"]
    assert matrix.scenario_ids() == ["trace-1", "trace-2"]
    assert len(matrix.outcomes) == 4  # 2 models x 2 scenarios x 1 episode
    outcome = matrix.for_scenario("trace-1")[0]
    assert outcome.model == "candidate-a"
    assert outcome.reward == pytest.approx(0.8)
    assert outcome.success is True
    assert outcome.critique == "fine"
    assert outcome.stop_reason == "agent_done"
    # Two policy calls (tool call + done), scripted usage 10in/5out each.
    assert outcome.usage == TokenUsage(input_tokens=20, output_tokens=10)
    assert len(outcome.call_seconds) == 2
    assert len(outcome.replies) == 2
    # Cost prices the POOL ENTRY's override, not the built-in table.
    assert outcome.cost_usd == pytest.approx((20 * 1.0 + 10 * 2.0) / 1_000_000)
    expensive = matrix.for_scenario("trace-1")[1]
    assert expensive.cost_usd == pytest.approx((20 * 10.0 + 10 * 20.0) / 1_000_000)


def test_evaluate_pool_repeats_episodes() -> None:
    matrix = evaluate_pool(
        lambda: _FakeEnv(EpisodeScore(reward=0.5, success=True, critique="")),
        _pool(),
        _SCENARIOS[:1],
        provider_factory=_ScriptedProvider,
        episodes_per_scenario=3,
    )
    episodes = [o.episode for o in matrix.outcomes if o.model == "candidate-a"]
    assert episodes == [0, 1, 2]


def test_evaluate_pool_requires_a_scoring_env() -> None:
    with pytest.raises(ValueError, match="score"):
        evaluate_pool(
            lambda: _FakeEnv(None),
            _pool(),
            _SCENARIOS[:1],
            provider_factory=_ScriptedProvider,
        )


def test_scenario_ids_fall_back_to_task_hash() -> None:
    matrix = evaluate_pool(
        lambda: _FakeEnv(EpisodeScore(reward=0.1, success=False, critique="")),
        _pool(),
        [Scenario(task="no provenance here")],
        provider_factory=_ScriptedProvider,
    )
    (scenario_id,) = matrix.scenario_ids()
    assert scenario_id  # deterministic, non-empty
    rerun = evaluate_pool(
        lambda: _FakeEnv(EpisodeScore(reward=0.1, success=False, critique="")),
        _pool(),
        [Scenario(task="no provenance here")],
        provider_factory=_ScriptedProvider,
    )
    assert rerun.scenario_ids() == [scenario_id]


def test_scoring_failure_costs_one_cell_not_the_sweep() -> None:
    # A raising last_score (throttled judge at close) leaves the cell unscored WITH the reason,
    # and every other cell of the sweep still gets measured.
    matrix = evaluate_pool(
        _ThrottledScoringEnv,
        _pool(),
        _SCENARIOS,
        provider_factory=_ScriptedProvider,
    )
    assert len(matrix.outcomes) == 4  # 2 models x 2 scenarios: nothing aborted
    for outcome in matrix.outcomes:
        assert outcome.reward is None
        assert outcome.success is False
        assert outcome.error is not None
        assert "ThrottlingException" in outcome.error
        assert outcome.steps == 1  # the episode itself ran fine; only scoring failed


def test_errored_episode_is_unscored_even_when_the_env_scored_it() -> None:
    # The agent died on its first call, so the env's score grades a run that never happened:
    # recording it as reward would read a provider throttle as incapability.
    matrix = evaluate_pool(
        lambda: _FakeEnv(EpisodeScore(reward=0.05, success=False, critique="did nothing")),
        _pool(),
        _SCENARIOS[:1],
        provider_factory=_FailFirstProvider,
    )
    outcome = matrix.outcomes[0]
    assert outcome.reward is None
    assert outcome.success is False
    assert outcome.error is not None and "429" in outcome.error
    # The judge text survives, labelled for what it is.
    assert outcome.critique == "salvage-judged despite error: did nothing"


def test_broken_progress_callback_does_not_abort_the_sweep() -> None:
    seen: list[str] = []

    def explode(outcome: ScenarioOutcome) -> None:
        seen.append(outcome.model)
        raise RuntimeError("progress socket closed")

    matrix = evaluate_pool(
        lambda: _FakeEnv(EpisodeScore(reward=0.8, success=True, critique="fine")),
        _pool(),
        _SCENARIOS,
        provider_factory=_ScriptedProvider,
        on_outcome=explode,
    )
    assert len(seen) == 4  # every cell still reported
    assert len(matrix.outcomes) == 4
    assert all(o.reward == pytest.approx(0.8) for o in matrix.outcomes)


def test_cells_run_candidate_minor_so_no_candidate_gets_a_cell_ahead() -> None:
    # The order invariant the blast radius rests on: scenario, then episode, then candidate.
    # Safe to pin because the sweep runs frozen, so no cell's predictions reach another cell.
    order: list[tuple[str, int, str]] = []
    evaluate_pool(
        lambda: _FakeEnv(EpisodeScore(reward=0.8, success=True, critique="fine")),
        _pool(),
        _SCENARIOS,
        episodes_per_scenario=2,
        provider_factory=_ScriptedProvider,
        on_outcome=lambda o: order.append((o.scenario_id, o.episode, o.model)),
    )
    assert order == [
        ("trace-1", 0, "candidate-a"),
        ("trace-1", 0, "candidate-b"),
        ("trace-1", 1, "candidate-a"),
        ("trace-1", 1, "candidate-b"),
        ("trace-2", 0, "candidate-a"),
        ("trace-2", 0, "candidate-b"),
        ("trace-2", 1, "candidate-a"),
        ("trace-2", 1, "candidate-b"),
    ]


def test_a_later_broken_candidate_surfaces_after_one_cell_per_candidate() -> None:
    """The blast radius of a backend that can only fail once it is CALLED.

    `candidate-c` is the Bedrock-with-no-credentials case: nothing local can see it, so it costs
    a cell. What must be bounded is how many cells the OTHER candidates burn before anyone
    watching `on_outcome` learns about it. Candidate-minor order makes that one cell each, and it
    does not depend on the grid: with candidate-major order the same pool hid the failure until
    cell 9 of 12, and any larger `--scenarios` would hide it for longer still.
    """
    pool = ModelPool(
        models=[
            *_pool().models,
            PoolEntry(
                name="candidate-c",
                kind=ProviderKind.BEDROCK,
                model="us.anthropic.claude-opus-4-8",
                input_per_mtok=15.0,
                output_per_mtok=75.0,
            ),
        ]
    )
    scenarios = [Scenario(task=f"task {i}", provenance=[f"trace-{i}"]) for i in range(4)]

    def _factory(entry: PoolEntry) -> _ScriptedProvider:
        if entry.name == "candidate-c":
            return _UnusableBackendProvider(entry)
        return _ScriptedProvider(entry)

    order: list[str] = []
    matrix = evaluate_pool(
        lambda: _FakeEnv(EpisodeScore(reward=0.8, success=True, critique="fine")),
        pool,
        scenarios,
        provider_factory=_factory,
        on_outcome=lambda o: order.append(o.model),
    )

    # Two cells ran before the broken candidate reported: one per earlier candidate.
    assert order.index("candidate-c") == 2
    # The sweep still completes, so the working candidates keep the evidence they were paid for
    # and the broken one's cells carry the diagnosis on every row.
    assert len(matrix.outcomes) == 12  # 3 candidates x 4 scenarios x 1 episode
    broken = [o for o in matrix.outcomes if o.model == "candidate-c"]
    assert len(broken) == 4
    assert all(not o.scored and "NoCredentialsError" in (o.error or "") for o in broken)
    assert all(o.scored for o in matrix.outcomes if o.model != "candidate-c")


def test_wrong_typed_score_yields_unscored_row_with_reason() -> None:
    # A last_score that isn't an EpisodeScore must not silently become an
    # unscored-with-no-error row: unscored rows always say why (outcomes contract).
    matrix = evaluate_pool(
        lambda: _FakeEnv(cast("EpisodeScore", {"reward": 1.0})),
        _pool(),
        _SCENARIOS[:1],
        provider_factory=_ScriptedProvider,
    )
    outcome = matrix.outcomes[0]
    assert outcome.reward is None
    assert outcome.error is not None and "not EpisodeScore" in outcome.error
