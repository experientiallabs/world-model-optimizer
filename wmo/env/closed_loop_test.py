"""Tests for the closed-loop pool evaluation (candidate models rolled against an Env)."""

from __future__ import annotations

import threading
import time
from typing import cast

import pytest

from wmo.core.types import Action, EnvState, Observation
from wmo.env.closed_loop import evaluate_pool
from wmo.env.scenarios import Scenario
from wmo.optimize.compression import CompressionConfig
from wmo.optimize.outcomes import ScenarioOutcome
from wmo.optimize.reward import EpisodeScore
from wmo.providers.base import (
    Completion,
    Message,
    ProviderConfig,
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


class _LongObservationEnv:
    """Env whose observations are far larger than any history budget under test."""

    last_score = EpisodeScore(reward=1.0, success=True, critique="")

    def reset(self, task: str | None = None, seed_state: EnvState | None = None) -> EnvState:
        return EnvState()

    def step(self, action: Action) -> Observation:
        return Observation(content="z" * 5000)

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
        return Completion(
            text=text,
            usage=TokenUsage(
                input_tokens=10,
                output_tokens=5,
                cached_input_tokens=3,
                cache_write_input_tokens=2,
                reasoning_tokens=4,
            ),
        )

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
    assert outcome.usage == TokenUsage(
        input_tokens=20,
        output_tokens=10,
        cached_input_tokens=6,
        cache_write_input_tokens=4,
        reasoning_tokens=8,
    )
    assert len(outcome.call_seconds) == 2
    assert outcome.call_input_tokens == [10, 10]
    assert outcome.call_output_tokens == [5, 5]
    assert outcome.call_cached_input_tokens == [3, 3]
    assert outcome.call_cache_write_input_tokens == [2, 2]
    assert outcome.tool_calls == 1
    assert len(outcome.replies) == 2
    # Cost prices the POOL ENTRY's override, not the built-in table.
    assert outcome.cost_usd == pytest.approx((10 * 1.0 + 6 * 1.0 + 4 * 1.0 + 10 * 2.0) / 1_000_000)
    expensive = matrix.for_scenario("trace-1")[1]
    assert expensive.cost_usd == pytest.approx(
        (10 * 10.0 + 6 * 10.0 + 4 * 10.0 + 10 * 20.0) / 1_000_000
    )


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


def test_history_chars_reaches_the_agent_from_evaluate_pool() -> None:
    """The knob must be real, not decorative.

    `history_chars` existed on `LLMAgent` but nothing that drives episodes could set it, so the
    only reachable value was the default. This pins the wiring: what `evaluate_pool` is given is
    what truncates the observation the agent sees on its next turn.
    """
    seen: list[str] = []

    class _Recorder:
        def __init__(self, entry: PoolEntry) -> None:
            self.config = ProviderConfig(kind=ProviderKind.ANTHROPIC, model=entry.model)

        def complete(
            self,
            system: str,
            messages: list[Message],
            *,
            temperature: float = 0.7,
            max_tokens: int = 8192,
        ) -> Completion:
            seen.append(messages[0].content)
            # Act once so there IS a later turn carrying an observation, then finish.
            if len(seen) == 1:
                return Completion(text='{"tool": "fetch", "arguments": {}}')
            return Completion(text='{"done": true}')

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] for _ in texts]

        def verify(self) -> VerifyResult:
            raise NotImplementedError

    evaluate_pool(
        lambda: _LongObservationEnv(),
        ModelPool(
            models=[
                PoolEntry(
                    name="m",
                    kind=ProviderKind.ANTHROPIC,
                    model="claude-x",
                    input_per_mtok=1.0,
                    output_per_mtok=1.0,
                )
            ]
        ),
        [Scenario(task="fetch the thing")],
        max_steps=2,
        history_chars=77,
        provider_factory=_Recorder,
    )
    later_turns = [prompt for prompt in seen if "EPISODE SO FAR" in prompt]
    assert later_turns, seen
    assert "z" * 77 in later_turns[0]
    assert "z" * 78 not in later_turns[0]


class _RecordingProvider(_ScriptedProvider):
    """Scripted provider that also records the messages each completion was asked with."""

    def __init__(self, entry: PoolEntry) -> None:
        super().__init__(entry)
        self.seen: list[list[Message]] = []

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        self.seen.append(list(messages))
        return super().complete(system, messages, temperature=temperature, max_tokens=max_tokens)


def test_evaluate_pool_threads_compression_through_the_measured_path() -> None:
    providers: list[_RecordingProvider] = []

    def factory(entry: PoolEntry) -> _RecordingProvider:
        provider = _RecordingProvider(entry)
        providers.append(provider)
        return provider

    matrix = evaluate_pool(
        lambda: _FakeEnv(EpisodeScore(reward=0.8, success=True, critique="fine")),
        _pool(),
        _SCENARIOS[:1],
        provider_factory=factory,
        max_steps=5,
        compression=CompressionConfig(compressor_id="truncate", aggressiveness=0.5),
    )

    outcome = matrix.outcomes[0]
    # The compressor's per-episode accounting landed on the row...
    assert outcome.compressor_id == "truncate"
    assert outcome.compressor_version == "1"
    assert outcome.aggressiveness == 0.5
    assert outcome.tokens_in_compressed < outcome.tokens_in_raw
    assert outcome.compressor_latency_s >= 0.0
    assert outcome.compressor_cost_usd == 0.0  # truncate has no inference cost
    # ...and the provider genuinely received compressed input (measured path, not
    # bookkeeping): the user content the model saw is strictly shorter than what an
    # uncompressed control run sends, and non-user content is untouched.
    control: list[_RecordingProvider] = []

    def control_factory(entry: PoolEntry) -> _RecordingProvider:
        provider = _RecordingProvider(entry)
        control.append(provider)
        return provider

    evaluate_pool(
        lambda: _FakeEnv(EpisodeScore(reward=0.8, success=True, critique="fine")),
        _pool(),
        _SCENARIOS[:1],
        provider_factory=control_factory,
        max_steps=5,
    )
    compressed_users = [m.content for m in providers[0].seen[0] if m.role == "user"]
    raw_users = [m.content for m in control[0].seen[0] if m.role == "user"]
    assert len(compressed_users) == len(raw_users)
    assert all(len(c) < len(r) for c, r in zip(compressed_users, raw_users, strict=True))


def test_uncompressed_rows_keep_default_compression_fields() -> None:
    matrix = evaluate_pool(
        lambda: _FakeEnv(EpisodeScore(reward=0.8, success=True, critique="fine")),
        _pool(),
        _SCENARIOS[:1],
        provider_factory=_ScriptedProvider,
        max_steps=5,
    )
    outcome = matrix.outcomes[0]
    assert outcome.compressor_id == ""
    assert outcome.tokens_in_raw == 0
    assert outcome.tokens_in_compressed == 0
    assert outcome.aggressiveness == 0.0


class _SleepingProvider(_ScriptedProvider):
    """Scripted provider whose every completion takes real wall time, like a real one does."""

    delay_s = 0.05

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        time.sleep(self.delay_s)
        return super().complete(system, messages, temperature=temperature, max_tokens=max_tokens)


def _timed_evaluation(max_concurrency: int) -> tuple[float, list[ScenarioOutcome]]:
    """Wall seconds to measure the 4-cell grid at `max_concurrency`, and the rows it produced."""
    started = time.monotonic()
    matrix = evaluate_pool(
        lambda: _FakeEnv(EpisodeScore(reward=0.8, success=True, critique="fine")),
        _pool(),
        _SCENARIOS,
        provider_factory=_SleepingProvider,
        max_steps=2,
        max_concurrency=max_concurrency,
    )
    return time.monotonic() - started, matrix.outcomes


def test_concurrent_cells_finish_in_a_fraction_of_the_sequential_wall_clock() -> None:
    """The point of the whole change: cells are IO-bound, so they overlap.

    4 cells x 2 completions x 50ms = 400ms of sleeping. Sequentially that is the wall clock;
    four at a time it is one cell's worth. The assertion is deliberately coarse (the ratio, plus
    a ceiling well above the ideal 100ms) so a loaded CI box cannot make it flap, while a
    regression that quietly serializes the pool again cannot pass it.
    """
    concurrent_s, concurrent_rows = _timed_evaluation(4)
    sequential_s, sequential_rows = _timed_evaluation(1)
    assert concurrent_s < 0.25, f"4 cells at once took {concurrent_s:.3f}s"
    assert sequential_s > concurrent_s * 1.8
    # Same evidence either way: concurrency is a speed knob, not a measurement change.
    assert [(row.model, row.scenario_id) for row in concurrent_rows] == [
        (row.model, row.scenario_id) for row in sequential_rows
    ]
    assert all(row.reward == pytest.approx(0.8) for row in concurrent_rows)


def test_the_matrix_is_in_cell_order_even_when_the_cells_finish_backwards() -> None:
    """Row order is the grid's order, not the finish order, or every digest of it drifts.

    The second candidate is made much faster than the first, so completion order is the reverse
    of cell order. The rows must still come back candidate-major.
    """
    finished: list[str] = []

    class _PacedProvider(_ScriptedProvider):
        def complete(
            self,
            system: str,
            messages: list[Message],
            *,
            temperature: float = 0.7,
            max_tokens: int = 8192,
        ) -> Completion:
            time.sleep(0.08 if self.config.model == "custom-a" else 0.0)
            return super().complete(
                system, messages, temperature=temperature, max_tokens=max_tokens
            )

    matrix = evaluate_pool(
        lambda: _FakeEnv(EpisodeScore(reward=0.8, success=True, critique="fine")),
        _pool(),
        _SCENARIOS,
        provider_factory=_PacedProvider,
        max_steps=2,
        max_concurrency=4,
        on_outcome=lambda outcome: finished.append(outcome.model),
    )
    assert finished[0] == "candidate-b"  # the fast candidate really did finish first
    assert [row.model for row in matrix.outcomes] == [
        "candidate-a",
        "candidate-a",
        "candidate-b",
        "candidate-b",
    ]


def test_the_progress_callback_never_runs_two_cells_at_once() -> None:
    # A console-printing callback that interleaved would corrupt the operator's progress lines,
    # so the callback is serialized even though the cells are not.
    inside = 0
    overlaps = 0

    def watch(outcome: ScenarioOutcome) -> None:
        nonlocal inside, overlaps
        inside += 1
        if inside > 1:
            overlaps += 1
        time.sleep(0.01)
        inside -= 1

    evaluate_pool(
        lambda: _FakeEnv(EpisodeScore(reward=0.8, success=True, critique="fine")),
        _pool(),
        _SCENARIOS,
        provider_factory=_ScriptedProvider,
        max_concurrency=4,
        on_outcome=watch,
    )
    assert overlaps == 0


def test_a_cell_that_raises_under_concurrency_still_reports_the_cells_that_finished() -> None:
    # The failure aborts the measurement (there is no matrix to return), but the cells that were
    # already paid for reached the callback, which is what the sweep persists them from.
    seen: list[str] = []

    class _ExplodingEnv(_FakeEnv):
        def reset(self, task: str | None = None, seed_state: EnvState | None = None) -> EnvState:
            raise RuntimeError("world model session refused")

    envs = iter(
        [
            _FakeEnv(EpisodeScore(reward=0.8, success=True, critique="ok")),
            _ExplodingEnv(None),
        ]
    )
    lock = threading.Lock()

    def next_env() -> _FakeEnv:
        with lock:
            return next(envs)

    with pytest.raises(RuntimeError, match="session refused"):
        evaluate_pool(
            next_env,
            ModelPool(models=[_pool().models[0]]),
            _SCENARIOS,
            provider_factory=_ScriptedProvider,
            max_concurrency=2,
            on_outcome=lambda outcome: seen.append(outcome.scenario_id),
        )
    assert len(seen) == 1


def test_a_concurrency_below_one_is_a_usage_error() -> None:
    with pytest.raises(ValueError, match="max_concurrency must be >= 1"):
        evaluate_pool(
            lambda: _FakeEnv(EpisodeScore(reward=0.1, success=False, critique="")),
            _pool(),
            _SCENARIOS[:1],
            provider_factory=_ScriptedProvider,
            max_concurrency=0,
        )


def test_pre_compression_outcome_json_loads_with_defaults() -> None:
    # Additive-fields guarantee for the matrix: rows serialized before the D-COMPRESS fields
    # existed must load with them defaulted.
    row = ScenarioOutcome(scenario_id="s", task="t", model="candidate-a")
    dumped = {
        key: value
        for key, value in row.model_dump(mode="json").items()
        if not key.startswith(("tokens_in_", "compressor_")) and key != "aggressiveness"
    }
    loaded = ScenarioOutcome.model_validate(dumped)
    assert loaded.compressor_id == ""
    assert loaded.tokens_in_raw == 0
    assert loaded.compressor_cost_usd == 0.0
