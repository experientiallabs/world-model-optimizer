"""End-to-end tests for exact, bounded, and resumable sandbox simulation."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Literal

import pytest

from exp.common.core.artifacts import ArtifactInput, canonical_json_bytes
from exp.common.evaluations import EvaluationCell, EvaluationPlan
from exp.common.models import (
    AssistantAction,
    BillingSource,
    ModelClient,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    NumericMeasurement,
    OperationEconomics,
    RoutedCandidateSnapshot,
    ToolCall,
    Usage,
)
from exp.common.project import ArtifactStore, ProjectPaths, artifact_input
from exp.common.rollouts import (
    RolloutArtifact,
    RolloutEventKind,
    SandboxSimulationCellBinding,
    SimulationMode,
    StopReason,
)
from exp.common.tasks import TaskCase, TaskSet
from exp.runtime.agents import AgentEpisode, AgentRuntime
from exp.runtime.environments import EnvironmentRuntime, EnvironmentSession, Observation
from exp.runtime.environments.harbor import (
    BOUNDED_CLEANUP_CONTRACT,
    HarborCleanupResult,
    HarborCommandResult,
    HarborEnvironmentRuntime,
)
from exp.runtime.environments.sandbox_ledger import read_ledger_files
from exp.simulation.engines.sandbox import (
    CandidateBinding,
    EnvironmentCostBinding,
    SandboxSimulationError,
    SandboxSimulator,
)
from exp.simulation.engines.sandbox_bindings import SandboxSimulationResolution
from exp.simulation.orchestration.interface import SimulationModeUnsupportedError
from exp.simulation.specs import MixedRealitySettings, SandboxSettings, SimulationSpec

_TIME = datetime(2026, 8, 12, tzinfo=UTC)
_ENVIRONMENT_DIGEST = "e" * 64


def test_sandbox_persists_each_rollout_and_resumes_without_reexecution(tmp_path: Path) -> None:
    """A completed cell is durable before the final set and an exact rerun performs no work."""
    store, plan, plan_input, task_input = _persist_fixture(tmp_path, ("task-a",))
    runtime = _EnvironmentRuntime()
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_input,
        runtime,
        agent_factory=_ToolAgent,
    )
    spec = _spec(plan_input, task_input, ("cell-a",))
    first = simulator.run(spec)
    rollout = _load_rollout(store, first.artifact_ids[0])
    resumed = simulator.run(spec)
    assert resumed == first
    assert runtime.opened_task_ids == ["task-a"]
    assert runtime.close_calls == 1
    assert rollout.stop_reason == StopReason.COMPLETED
    assert rollout.sandbox_binding is not None
    assert rollout.sandbox_binding.environment_sha256 == _ENVIRONMENT_DIGEST
    assert rollout.sandbox_binding.task_lineage_group_id == "lineage-task-a"
    kinds = {span.kind for span in rollout.spans}
    assert kinds.issuperset({RolloutEventKind.TOOL_CALL, RolloutEventKind.OBSERVATION})
    assert rollout.candidate_economics.cost_usd is None
    assert rollout.sandbox_economics is not None
    assert rollout.sandbox_economics.cost_usd is None
    assert store.read(first.artifact_set_id).manifest.artifact_type == ("simulation-artifact-set")


def test_sparse_fit_and_held_out_cells_have_distinct_exact_bindings(tmp_path: Path) -> None:
    """Purpose and cell ID prevent legal sparse cells from colliding at resolution or rollout."""
    cells = (
        EvaluationCell(
            cell_id="cell-fit",
            task_id="task-a",
            candidate_alias="candidate-a",
            repeat=0,
            purpose="fit",
            execution="simulate",
        ),
        EvaluationCell(
            cell_id="cell-held",
            task_id="task-a",
            candidate_alias="candidate-a",
            repeat=0,
            purpose="held_out",
            execution="simulate",
        ),
    )
    store, plan, plan_input, task_input = _persist_fixture(
        tmp_path,
        ("task-a",),
        cells=cells,
    )
    artifact_set = _simulator(
        store,
        plan,
        plan_input,
        task_input,
        _EnvironmentRuntime(),
        agent_factory=_ToolAgent,
    ).run(_spec(plan_input, task_input, ("cell-fit", "cell-held")))
    rollouts = tuple(_load_rollout(store, item) for item in artifact_set.artifact_ids)
    assert rollouts[0].rollout_id != rollouts[1].rollout_id
    assert tuple(item.sandbox_binding.purpose for item in rollouts if item.sandbox_binding) == (
        "fit",
        "held_out",
    )
    resolution_id = _artifact_ids_of_type(store, "sandbox-simulation-resolution")[0]
    resolution = SandboxSimulationResolution.model_validate_json(
        store.read_bytes(resolution_id, "sandbox-simulation-resolution.json")
    )
    duplicate_payload = resolution.model_dump(mode="json")
    duplicate_payload["cell_bindings"] = [
        resolution.cell_bindings[0].model_dump(mode="json"),
        resolution.cell_bindings[0].model_dump(mode="json"),
    ]
    with pytest.raises(ValueError, match="cell IDs must be unique"):
        SandboxSimulationResolution.model_validate(duplicate_payload)


def test_sandbox_rejects_unplanned_observed_mixed_and_environment_mismatch(
    tmp_path: Path,
) -> None:
    """Every unsupported or mismatched selection fails before a customer environment opens."""
    store, plan, plan_input, task_input = _persist_fixture(
        tmp_path,
        ("task-a",),
        execution="observed",
    )
    runtime = _EnvironmentRuntime()
    simulator = _simulator(store, plan, plan_input, task_input, runtime)

    with pytest.raises(SandboxSimulationError, match="observed cell"):
        simulator.run(_spec(plan_input, task_input, ("cell-a",)))
    with pytest.raises(SandboxSimulationError, match="outside its evaluation plan"):
        simulator.run(_spec(plan_input, task_input, ("cell-missing",)))
    with pytest.raises(SandboxSimulationError, match="environment identity"):
        simulator.run(
            _spec(
                plan_input,
                task_input,
                ("cell-a",),
                environment_sha256="f" * 64,
            )
        )
    mixed = SimulationSpec(
        schema_version=1,
        created_at=_TIME,
        inputs=(plan_input, task_input),
        code_revision="test-revision",
        simulation_id="simulation-mixed",
        evaluation_plan_id=plan.plan_id,
        cell_ids=("cell-a",),
        agent_id="customer-agent",
        mode=SimulationMode.MIXED_REALITY,
        mixed_reality=MixedRealitySettings(policy_id="future-policy"),
        seed=7,
        maximum_steps=2,
    )
    with pytest.raises(SimulationModeUnsupportedError, match="not implemented"):
        simulator.run(mixed)

    assert runtime.opened_task_ids == []


def test_maximum_steps_is_a_hard_candidate_dispatch_limit(tmp_path: Path) -> None:
    """The recorder admits exactly the configured number of calls and reports the stop reason."""
    store, plan, plan_input, task_input = _persist_fixture(tmp_path, ("task-a",))
    runtime = _EnvironmentRuntime()
    client = _ScriptedClient([0.1, 0.1])
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_input,
        runtime,
        client=client,
        agent_factory=lambda: _LoopingAgent(3),
    )

    artifact_set = simulator.run(_spec(plan_input, task_input, ("cell-a",), maximum_steps=2))
    rollout = _load_rollout(store, artifact_set.artifact_ids[0])

    assert len(client.requests) == 2
    assert rollout.stop_reason == StopReason.MAXIMUM_STEPS
    assert rollout.failure is not None
    assert rollout.failure.exception_type == "SandboxStepLimitError"
    assert runtime.close_calls == 1


def test_maximum_time_interrupts_a_silent_agent_and_still_cleans_up(tmp_path: Path) -> None:
    """A non-cooperative agent cannot evade the wall bound, and cleanup still runs once."""
    store, plan, plan_input, task_input = _persist_fixture(tmp_path, ("task-a",))
    runtime = _EnvironmentRuntime()
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_input,
        runtime,
        agent_factory=lambda: _SleepingAgent(2.0),
    )
    started = time.monotonic()

    # The wall bound must outlast pre-episode setup on a loaded runner: _Deadline starts
    # at construction, and an expired deadline raises before the environment ever opens,
    # which would make close_calls 0 without any cleanup having been skipped. Interrupting
    # the 2.0s sleep is proven by finishing well under 2.0s, not by a tiny bound.
    artifact_set = simulator.run(
        _spec(plan_input, task_input, ("cell-a",), maximum_time_seconds=0.5)
    )
    rollout = _load_rollout(store, artifact_set.artifact_ids[0])

    assert time.monotonic() - started < 1.5
    assert rollout.stop_reason == StopReason.MAXIMUM_TIME
    assert rollout.failure is not None
    assert rollout.failure.exception_type == "SandboxTimeLimitError"
    assert runtime.close_calls == 1


def test_permanently_hung_cleanup_returns_bounded_failure(tmp_path: Path) -> None:
    """The hard episode timer interrupts cleanup and never reports the cell as completed."""
    store, plan, plan_input, task_input = _persist_fixture(tmp_path, ("task-a",))
    runtime = _EnvironmentRuntime(hang_cleanup=True)
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_input,
        runtime,
        agent_factory=_ToolAgent,
    )
    started = time.monotonic()
    artifact_set = simulator.run(
        _spec(plan_input, task_input, ("cell-a",), maximum_time_seconds=0.2)
    )
    elapsed = time.monotonic() - started
    rollout = _load_rollout(store, artifact_set.artifact_ids[0])

    # The bound proves an infinitely hung cleanup was interrupted. Only the
    # timed run is measured so artifact loading cannot mask a slow interrupt,
    # and the bound stays tight enough to catch a multi-second regression
    # while tolerating loaded-runner overhead around the short 0.2s timer.
    assert elapsed < 2.0
    assert rollout.stop_reason == StopReason.MAXIMUM_TIME
    assert rollout.failure is not None
    assert rollout.failure.attribution is not None
    assert rollout.failure.attribution.value == "cleanup"
    assert rollout.failure.details["phase"] == "cleanup_timeout"
    assert runtime.close_calls == 1


def test_finite_cost_is_preflighted_reserved_and_never_fabricated(tmp_path: Path) -> None:
    """Finite spend needs a capability proof and stops before a request could exceed its cap."""
    store, plan, plan_input, task_input = _persist_fixture(tmp_path, ("task-a",))
    runtime = _EnvironmentRuntime()
    unsafe = _simulator(
        store,
        plan,
        plan_input,
        task_input,
        runtime,
        agent_factory=lambda: _LoopingAgent(2),
    )
    spec = _spec(plan_input, task_input, ("cell-a",), maximum_cost_usd=1.0)

    with pytest.raises(SandboxSimulationError, match="observable candidate cost"):
        unsafe.run(spec)
    assert runtime.opened_task_ids == []

    unsafe_environment = _simulator(
        store,
        plan,
        plan_input,
        task_input,
        runtime,
        client=_ScriptedClient([]),
        agent_factory=lambda: _LoopingAgent(2),
        maximum_call_cost_usd=0.7,
        cost_is_observable=True,
    )
    with pytest.raises(SandboxSimulationError, match="observable environment cost"):
        unsafe_environment.run(spec)
    assert runtime.opened_task_ids == []

    client = _ScriptedClient([0.4])
    bounded = _simulator(
        store,
        plan,
        plan_input,
        task_input,
        runtime,
        client=client,
        agent_factory=lambda: _LoopingAgent(2),
        maximum_call_cost_usd=0.7,
        cost_is_observable=True,
        environment_cost=EnvironmentCostBinding(
            maximum_episode_cost_usd=0,
            cost_is_observable=True,
        ),
    )
    artifact_set = bounded.run(spec)
    rollout = _load_rollout(store, artifact_set.artifact_ids[0])

    assert len(client.requests) == 1
    assert rollout.stop_reason == StopReason.MAXIMUM_COST
    assert rollout.candidate_economics.cost_usd is not None
    assert rollout.candidate_economics.cost_usd.value == pytest.approx(0.4)


def test_unknown_dispatched_cost_fails_closed_and_blocks_later_cells(tmp_path: Path) -> None:
    """In stop mode an unpriced dispatch is not zero spend, so no later paid cell runs."""
    store, plan, plan_input, task_input = _persist_fixture(
        tmp_path,
        ("task-a", "task-b"),
    )
    runtime = _EnvironmentRuntime()
    client = _ScriptedClient([None])
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_input,
        runtime,
        client=client,
        agent_factory=_OneCallAgent,
        maximum_call_cost_usd=0.5,
        cost_is_observable=True,
        environment_cost=EnvironmentCostBinding(
            maximum_episode_cost_usd=0,
            cost_is_observable=True,
        ),
    )

    artifact_set = simulator.run(
        _spec(
            plan_input,
            task_input,
            ("cell-a", "cell-b"),
            maximum_cost_usd=1.0,
            stop_on_overspend=True,
        )
    )
    first, second = tuple(_load_rollout(store, item) for item in artifact_set.artifact_ids)

    assert len(client.requests) == 1
    assert runtime.opened_task_ids == ["task-a"]
    assert first.stop_reason == StopReason.MAXIMUM_COST
    assert first.candidate_economics.cost_usd is None
    assert first.failure is not None
    assert first.failure.details["provider_dispatch_unknown_spend"] is True
    assert second.stop_reason == StopReason.MAXIMUM_COST
    assert second.failure is not None
    assert second.failure.details["observed_spend_usd"] is None


def test_unknown_environment_cost_fails_closed_and_blocks_later_cells(tmp_path: Path) -> None:
    """In stop mode an environment that omits promised cost blocks every later paid cell."""
    store, plan, plan_input, task_input = _persist_fixture(
        tmp_path,
        ("task-a", "task-b"),
    )
    runtime = _EnvironmentRuntime()
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_input,
        runtime,
        agent_factory=_ToolAgent,
        maximum_call_cost_usd=0.5,
        cost_is_observable=True,
        environment_cost=EnvironmentCostBinding(
            maximum_episode_cost_usd=0.25,
            cost_is_observable=True,
        ),
    )

    artifact_set = simulator.run(
        _spec(
            plan_input,
            task_input,
            ("cell-a", "cell-b"),
            maximum_cost_usd=1.0,
            stop_on_overspend=True,
        )
    )
    first, second = tuple(_load_rollout(store, item) for item in artifact_set.artifact_ids)

    assert runtime.opened_task_ids == ["task-a"]
    assert first.stop_reason == StopReason.MAXIMUM_COST
    assert first.sandbox_economics is not None
    assert first.sandbox_economics.cost_usd is None
    assert first.failure is not None
    assert first.failure.details["environment_dispatch_unknown_spend"] is True
    assert second.stop_reason == StopReason.MAXIMUM_COST
    assert second.failure is not None
    assert second.failure.details["observed_spend_usd"] is None


def test_observed_environment_cost_is_recorded_and_stays_inside_its_reservation(
    tmp_path: Path,
) -> None:
    """A nonzero environment cost remains separate, observed, and budget-admissible."""
    store, plan, plan_input, task_input = _persist_fixture(tmp_path, ("task-a",))
    runtime = _EnvironmentRuntime(cost_usd=0.2)
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_input,
        runtime,
        client=_ScriptedClient([]),
        agent_factory=_ToolAgent,
        maximum_call_cost_usd=0.5,
        cost_is_observable=True,
        environment_cost=EnvironmentCostBinding(
            maximum_episode_cost_usd=0.3,
            cost_is_observable=True,
        ),
    )

    artifact_set = simulator.run(_spec(plan_input, task_input, ("cell-a",), maximum_cost_usd=1.0))
    rollout = _load_rollout(store, artifact_set.artifact_ids[0])

    assert rollout.stop_reason == StopReason.COMPLETED
    assert rollout.sandbox_economics is not None
    assert rollout.sandbox_economics.cost_usd is not None
    assert rollout.sandbox_economics.cost_usd.value == pytest.approx(0.2)
    assert rollout.candidate_economics.cost_usd is None


def test_completed_cell_survives_a_crash_between_episode_boundaries(tmp_path: Path) -> None:
    """A process crash after cell A cannot replay A when a new simulator resumes cell B."""
    store, plan, plan_input, task_input = _persist_fixture(
        tmp_path,
        ("task-a", "task-b"),
    )
    runtime = _EnvironmentRuntime()
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_input,
        runtime,
        agent_factory=_ToolAgent,
    )
    original = simulator._execute_and_persist_cell

    def crash_before_second_cell(
        spec: SimulationSpec,
        cell: EvaluationCell,
        resolution: SandboxSimulationResolution,
        resolution_input: ArtifactInput,
        binding: SandboxSimulationCellBinding,
    ) -> RolloutArtifact:
        if cell.cell_id == "cell-b":
            raise KeyboardInterrupt("simulated process crash")
        return original(spec, cell, resolution, resolution_input, binding)

    simulator.__dict__["_execute_and_persist_cell"] = crash_before_second_cell
    spec = _spec(plan_input, task_input, ("cell-a", "cell-b"))
    with pytest.raises(KeyboardInterrupt, match="simulated process crash"):
        simulator.run(spec)

    assert len(_artifact_ids_of_type(store, "rollout")) == 1
    assert _artifact_ids_of_type(store, "simulation-artifact-set") == ()

    resumed = _simulator(
        store,
        plan,
        plan_input,
        task_input,
        runtime,
        agent_factory=_ToolAgent,
    ).run(spec)

    assert len(resumed.artifact_ids) == 2
    assert runtime.opened_task_ids == ["task-a", "task-b"]
    assert runtime.close_calls == 2


@pytest.mark.parametrize(
    ("hang_cleanup", "expected_stop", "expected_phase"),
    (
        (False, StopReason.FAILURE, "cleanup_failure"),
        (True, StopReason.MAXIMUM_TIME, "cleanup_timeout"),
    ),
)
def test_harbor_cleanup_failure_or_hang_holds_ledger_and_partial_transcript(
    tmp_path: Path,
    hang_cleanup: bool,
    expected_stop: StopReason,
    expected_phase: str,
) -> None:
    """Harbor evidence survives deterministic bounded cleanup failure and stays reaper-owned.

    The Harbor double reports its own bounded cleanup result, so the episode receives ordinary
    runtime headroom instead of racing all setup and command work against a 30 ms wall clock.
    """
    store, plan, plan_input, task_input = _persist_fixture(tmp_path, ("task-a",))
    backend = _HarborSession(hang_cleanup=hang_cleanup)
    state_directory = tmp_path / "harbor-state"
    harbor = HarborEnvironmentRuntime(
        _HarborFactory(backend),
        environment_id="customer-environment",
        template_name="exp-hb-v1-fixture",
        state_directory=state_directory,
        retry_delays_seconds=(),
    )
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_input,
        harbor,
        agent_factory=_ReadFileAgent,
    )
    started = time.monotonic()
    artifact_set = simulator.run(
        _spec(
            plan_input,
            task_input,
            ("cell-a",),
            maximum_time_seconds=30.0,
        )
    )
    rollout = _load_rollout(store, artifact_set.artifact_ids[0])
    observations = tuple(
        span for span in rollout.spans if span.kind == RolloutEventKind.OBSERVATION
    )
    assert time.monotonic() - started < 1.0
    assert rollout.stop_reason == expected_stop
    assert rollout.failure is not None
    assert rollout.failure.attribution is not None
    assert rollout.failure.attribution.value == "cleanup"
    assert rollout.failure.details["phase"] == expected_phase
    assert observations[0].payload["content"] == "partial output"
    assert observations[0].failure is not None
    ledger_state = read_ledger_files(state_directory)[0]
    assert tuple(item.sandbox_id for item in ledger_state.held) == ("sandbox-harbor",)
    assert ledger_state.released_ids == ()


def _persist_fixture(
    root: Path,
    task_ids: tuple[str, ...],
    *,
    execution: Literal["observed", "simulate"] = "simulate",
    cells: tuple[EvaluationCell, ...] | None = None,
) -> tuple[ArtifactStore, EvaluationPlan, ArtifactInput, ArtifactInput]:
    """Persist one exact task set and evaluation plan for sandbox tests."""
    store = ArtifactStore(ProjectPaths(root=root, project_id="project-a"))
    tasks = tuple(_task(task_id) for task_id in task_ids)
    task_payload = b"\n".join(canonical_json_bytes(task) for task in tasks) + b"\n"
    task_set = TaskSet(
        schema_version=1,
        created_at=_TIME,
        code_revision="test-revision",
        task_set_id="task-set-1",
        task_ids=task_ids,
        tasks_path="tasks.jsonl",
        tasks_sha256=hashlib.sha256(task_payload).hexdigest(),
    )
    task_manifest = store.write(
        artifact_id=task_set.task_set_id,
        artifact_type="task-set",
        envelope=task_set,
        files={"task-set.json": canonical_json_bytes(task_set), "tasks.jsonl": task_payload},
    )
    selected_cells = cells or tuple(
        _cell(index, task_id, execution=execution) for index, task_id in enumerate(task_ids)
    )
    plan = EvaluationPlan(
        schema_version=2,
        created_at=_TIME,
        code_revision="test-revision",
        plan_id="plan-1",
        task_set_id=task_set.task_set_id,
        candidate_snapshots=(RoutedCandidateSnapshot(alias="candidate-a", model=_snapshot()),),
        pricing_snapshot_id="pricing-1",
        pricing_snapshot_sha256="d" * 64,
        cells=selected_cells,
    )
    plan_manifest = store.write_json(
        artifact_id=plan.plan_id,
        artifact_type="evaluation-plan",
        envelope=plan,
        files={"evaluation-plan.json": plan},
    )
    return store, plan, artifact_input(plan_manifest), artifact_input(task_manifest)


def _simulator(
    store: ArtifactStore,
    plan: EvaluationPlan,
    plan_input: ArtifactInput,
    task_input: ArtifactInput,
    runtime: EnvironmentRuntime,
    *,
    client: ModelClient | None = None,
    agent_factory: Callable[[], AgentRuntime] | None = None,
    maximum_call_cost_usd: float | None = None,
    cost_is_observable: bool = False,
    environment_cost: EnvironmentCostBinding | None = None,
) -> SandboxSimulator:
    """Build one fully local sandbox simulator with no provider or network access."""
    return SandboxSimulator(
        store=store,
        evaluation_plan=plan,
        evaluation_plan_input=plan_input,
        task_set_input=task_input,
        candidates={
            "candidate-a": CandidateBinding(
                alias="candidate-a",
                client=client or _UnexpectedClient(),
                snapshot=_snapshot(),
                maximum_call_cost_usd=maximum_call_cost_usd,
                cost_is_observable=cost_is_observable,
            )
        },
        agent_factory=agent_factory or _OneCallAgent,
        environment_runtime=runtime,
        environment_cost=environment_cost,
        environment_id="customer-environment",
        environment_sha256=_ENVIRONMENT_DIGEST,
        source_run_id="sandbox-run-1",
        clock=lambda: _TIME,
    )


def _spec(
    plan_input: ArtifactInput,
    task_input: ArtifactInput,
    cell_ids: tuple[str, ...],
    *,
    maximum_steps: int = 3,
    maximum_time_seconds: float = 30.0,
    maximum_cost_usd: float | None = None,
    environment_sha256: str = _ENVIRONMENT_DIGEST,
    stop_on_overspend: bool = False,
) -> SimulationSpec:
    """Create one exact shared W8 sandbox specification."""
    return SimulationSpec(
        schema_version=1,
        created_at=_TIME,
        inputs=(plan_input, task_input),
        code_revision="test-revision",
        simulation_id="simulation-1",
        evaluation_plan_id="plan-1",
        cell_ids=cell_ids,
        agent_id="customer-agent",
        mode=SimulationMode.SANDBOX,
        sandbox=SandboxSettings(
            environment_id="customer-environment",
            environment_sha256=environment_sha256,
            maximum_time_seconds=maximum_time_seconds,
        ),
        seed=7,
        maximum_steps=maximum_steps,
        maximum_cost_usd=maximum_cost_usd,
        stop_on_overspend=stop_on_overspend,
    )


def _cell(
    index: int,
    task_id: str,
    *,
    execution: Literal["observed", "simulate"],
) -> EvaluationCell:
    """Create one explicit held-out cell with an optional observed marker."""
    suffix = chr(ord("a") + index)
    return EvaluationCell(
        cell_id=f"cell-{suffix}",
        task_id=task_id,
        candidate_alias="candidate-a",
        repeat=0,
        purpose="held_out",
        execution=execution,
        observed_rollout_id="production-rollout-1" if execution == "observed" else None,
    )


def _task(task_id: str) -> TaskCase:
    """Create one held-out executable task with stable lineage."""
    return TaskCase(
        task_id=task_id,
        lineage_group_id=f"lineage-{task_id}",
        partition="held_out",
        instruction=f"Complete {task_id} in the environment.",
        workload_weight=1.0,
        source_trace_ids=(f"trace-{task_id}",),
    )


def _snapshot() -> ModelSnapshot:
    """Return the pinned candidate identity shared by plan and runtime."""
    return ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        provider="fixture",
        model_id="candidate-a",
        revision="v1",
        capabilities_sha256="a" * 64,
        connection_sha256="b" * 64,
    )


def _load_rollout(store: ArtifactStore, rollout_id: str) -> RolloutArtifact:
    """Load one digest-verified canonical rollout."""
    return RolloutArtifact.model_validate_json(store.read_bytes(rollout_id, "rollout.json"))


def _artifact_ids_of_type(store: ArtifactStore, artifact_type: str) -> tuple[str, ...]:
    """Return sorted stored IDs for one manifest type."""
    return tuple(
        artifact_id
        for artifact_id in store.list_ids()
        if store.read(artifact_id).manifest.artifact_type == artifact_type
    )


class _UnexpectedClient:
    """Fail if a tool-only or sleeping fixture unexpectedly dispatches a model request."""

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Reject an unexpected provider-shaped call."""
        raise AssertionError(f"unexpected model request: {request!r}")


class _ScriptedClient:
    """Return local responses with explicitly present or absent cost evidence."""

    def __init__(self, costs: list[float | None]) -> None:
        self._costs = list(costs)
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Record one request and return its scripted economics."""
        self.requests.append(request)
        cost = self._costs.pop(0)
        return ModelResponse(
            output=AssistantAction(content="candidate response"),
            model=_snapshot(),
            economics=OperationEconomics(
                usage=Usage(input_tokens=4, output_tokens=2),
                cost_usd=(
                    NumericMeasurement(value=cost, provenance="observed")
                    if cost is not None
                    else None
                ),
            ),
        )


def _request(task: TaskCase) -> ModelRequest:
    """Build a minimal candidate request from visible task text."""
    return ModelRequest(messages=(ModelMessage(role="user", content=task.instruction),))


class _OneCallAgent:
    """Make one candidate request and return its visible action."""

    def run(
        self,
        task: TaskCase,
        *,
        model: ModelClient,
        environment: EnvironmentSession,
    ) -> AgentEpisode:
        """Execute exactly one injected model call."""
        del environment
        response = model.complete(_request(task))
        return AgentEpisode(stop_reason=StopReason.COMPLETED, final_action=response.output)


class _LoopingAgent:
    """Attempt a fixed number of candidate calls so simulator limits can interrupt it."""

    def __init__(self, calls: int) -> None:
        self._calls = calls

    def run(
        self,
        task: TaskCase,
        *,
        model: ModelClient,
        environment: EnvironmentSession,
    ) -> AgentEpisode:
        """Return only when every requested candidate call was admitted."""
        del environment
        output: AssistantAction | None = None
        for _ in range(self._calls):
            output = model.complete(_request(task)).output
        return AgentEpisode(stop_reason=StopReason.COMPLETED, final_action=output)


class _ToolAgent:
    """Use one executable tool without dispatching a candidate model request."""

    def run(
        self,
        task: TaskCase,
        *,
        model: ModelClient,
        environment: EnvironmentSession,
    ) -> AgentEpisode:
        """Record one successful tool result in the canonical transcript."""
        del task, model
        observation = environment.execute(ToolCall(call_id="call-1", name="lookup"))
        assert observation.content == "environment output"
        return AgentEpisode(
            stop_reason=StopReason.COMPLETED,
            final_action=AssistantAction(content="completed"),
        )


class _ReadFileAgent:
    """Exercise the retained Harbor read adapter and ignore its expected error observation."""

    def run(
        self,
        task: TaskCase,
        *,
        model: ModelClient,
        environment: EnvironmentSession,
    ) -> AgentEpisode:
        """Return normally so the later Harbor cleanup proof is the terminal failure."""
        del task, model
        environment.execute(
            ToolCall(call_id="call-read", name="read_file", arguments={"path": "notes.txt"})
        )
        return AgentEpisode(
            stop_reason=StopReason.COMPLETED,
            final_action=AssistantAction(content="saw partial output"),
        )


class _SleepingAgent:
    """Block without model or tool calls to exercise the hard episode timer."""

    def __init__(self, duration_seconds: float) -> None:
        self._duration_seconds = duration_seconds

    def run(
        self,
        task: TaskCase,
        *,
        model: ModelClient,
        environment: EnvironmentSession,
    ) -> AgentEpisode:
        """Sleep longer than the configured simulator wall bound."""
        del task, model, environment
        time.sleep(self._duration_seconds)
        return AgentEpisode(stop_reason=StopReason.COMPLETED)


class _EnvironmentRuntime:
    """Provide fresh deterministic sessions and record every cleanup boundary."""

    def __init__(
        self,
        *,
        cost_usd: float | None = None,
        hang_cleanup: bool = False,
    ) -> None:
        self.opened_task_ids: list[str] = []
        self.close_calls = 0
        self.cost_usd = cost_usd
        self.hang_cleanup = hang_cleanup

    def open(self, task: TaskCase) -> AbstractContextManager[EnvironmentSession]:
        """Open one task-bound in-memory environment."""
        self.opened_task_ids.append(task.task_id)
        return _EnvironmentContext(self)


class _EnvironmentContext(AbstractContextManager[EnvironmentSession]):
    """Record context exit as direct cleanup evidence."""

    def __init__(self, runtime: _EnvironmentRuntime) -> None:
        self._runtime = runtime

    def __enter__(self) -> EnvironmentSession:
        """Return a fresh execute-only backing session."""
        return _EnvironmentSession(self._runtime.cost_usd)

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Record cleanup on every normal and exceptional path."""
        del exception_type, exception, traceback
        self._runtime.close_calls += 1
        if self._runtime.hang_cleanup:
            while True:
                pass
        return False


class _EnvironmentSession:
    """Return one local observation for the sandbox tool fixture."""

    def __init__(self, cost_usd: float | None) -> None:
        self._cost_usd = cost_usd

    def execute(self, action: ToolCall) -> Observation:
        """Execute the fixture lookup action."""
        assert action.name == "lookup"
        economics = OperationEconomics(
            cost_usd=(
                NumericMeasurement(value=self._cost_usd, provenance="observed")
                if self._cost_usd is not None
                else None
            )
        )
        return Observation(
            content="environment output",
            metadata={"economics": economics.model_dump(mode="json")},
        )


class _HarborFactory:
    """Open one deterministic Harbor-shaped backend session."""

    cleanup_contract: Literal["bounded-close-v1"] = BOUNDED_CLEANUP_CONTRACT

    def __init__(self, session: _HarborSession) -> None:
        self._session = session

    def open(
        self,
        task: TaskCase,
        *,
        template_name: str,
    ) -> _HarborSession:
        """Bind the expected task and frozen template identity."""
        assert task.task_id == "task-a"
        assert template_name == "exp-hb-v1-fixture"
        return self._session


class _HarborSession:
    """Return partial command evidence and reject final cleanup verification."""

    sandbox_id = "sandbox-harbor"
    template_id = "exp-hb-v1-fixture"

    def __init__(self, *, hang_cleanup: bool = False) -> None:
        self.closed = False
        self.hang_cleanup = hang_cleanup
        self.commands: list[tuple[str, Mapping[str, str] | None]] = []

    def execute_command(
        self,
        command: str,
        *,
        environment: Mapping[str, str] | None,
        timeout_seconds: int,
    ) -> HarborCommandResult:
        """Return one failed command with usable stdout and explicit unknown cost."""
        del timeout_seconds
        self.commands.append((command, environment))
        return HarborCommandResult(stdout="partial output", exit_code=7)

    def close(self, *, sandbox_id: str, timeout_seconds: float) -> HarborCleanupResult:
        """Return deterministic bounded timeout or unproven cleanup evidence."""
        assert sandbox_id == "sandbox-harbor"
        assert timeout_seconds > 0
        self.closed = not self.hang_cleanup
        return HarborCleanupResult(
            sandbox_id=sandbox_id,
            released=False,
            timed_out=self.hang_cleanup,
            failure=None if self.hang_cleanup else "cleanup was not proven",
        )
