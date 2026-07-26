"""Evaluator-neutral score evidence for harness optimization.

A `Scorer` runs one exact `HarnessDoc` candidate against a fixed task-by-attempt matrix and
returns a `ScoreReport` of raw evaluator rewards plus their pass interpretation. The contract is
deliberately small: the optimizer loop (PR C) ranks candidates by `ScoreReport.score` and reads
per-trial artifacts straight from each cell's `artifact_dir`; the evaluator's own output
directory is the record, so the report carries paths, not copied or hashed evidence.

Reward interpretation is frozen protocol (paper semantics): `positive-binary` counts a trial as
passed iff its raw reward is strictly positive; `raw` counts it passed iff the reward is exactly
1.0. Raw rewards stay untouched in the cells either way.
"""

from __future__ import annotations

from collections.abc import Callable
from statistics import fmean, median
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from wmh.core.text import validate_durable_text
from wmh.harness.doc import HarnessDoc

RewardMode = Literal["raw", "positive-binary"]

_DOC_HASH_PATTERN = r"^[0-9a-f]{32}$"
MAX_CELL_NOTE_CHARS = 2_000


def reward_passed(reward: float, mode: RewardMode) -> bool:
    """Interpret one raw evaluator reward under the frozen selection protocol."""
    if mode == "raw":
        return reward == 1.0
    return reward > 0.0


def _p90(values: list[int]) -> int:
    """The 90th-percentile value by nearest-rank (deterministic, no interpolation)."""
    ordered = sorted(values)
    # Nearest-rank: rank ceil(0.9 * n), 1-indexed, clamped into the sample.
    rank = -(-9 * len(ordered) // 10)  # ceil(0.9 * n) via ceil-division
    return ordered[min(max(rank, 1), len(ordered)) - 1]


class ScoreRequest(BaseModel):
    """One exact task-by-attempt matrix a scorer evaluates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_ids: tuple[str, ...]
    attempts: int = Field(strict=True, ge=1)

    @field_validator("attempts", mode="before")
    @classmethod
    def _reject_boolean_attempts(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("attempts must be an integer, not boolean")
        return value

    @field_validator("task_ids")
    @classmethod
    def _validate_task_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("task_ids must be nonempty")
        if len(set(value)) != len(value):
            raise ValueError("task_ids must be unique")
        for task_id in value:
            if not task_id:
                raise ValueError("task_ids must not contain empty values")
            validate_durable_text(task_id, field="task id")
        return value


class ScoreCell(BaseModel):
    """One trial: the raw evaluator reward, its pass interpretation, and where the evidence is."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    task_id: str = Field(strict=True, min_length=1)
    attempt: int = Field(strict=True, ge=1)
    # The evaluator's raw reward, untouched. `passed` applies the report's reward mode.
    reward: float = Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)
    passed: bool = Field(strict=True)
    # The evaluator-owned directory holding this trial's raw evidence (config/result/logs).
    # May be empty for scorers that keep no per-trial directory.
    artifact_dir: str = ""
    # A short diagnostic, e.g. "completed with AgentTimeoutError". A trial that failed but still
    # produced a verifier reward is a candidate outcome, and the note says why it looks that way.
    note: str = Field(default="", max_length=MAX_CELL_NOTE_CHARS)
    # Optional per-episode token telemetry lifted from the trial's WMH transcript. All fields are
    # best-effort and default when no transcript exists or it cannot be parsed, so every existing
    # construction and the `.score`/`.pass_rate` contract stay unchanged. The proposer reads these
    # to tell which harness settings (turn cap, output cap, context size) actually change behavior.
    turns: int | None = None
    calls: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    # The episode's stop reason (e.g. "submitted", "max_turns", "cancelled-by-harbor-timeout").
    stop_reason: str = Field(default="", max_length=MAX_CELL_NOTE_CHARS)
    # True when the episode reached the scored doc's turn cap (a lever raising max_turns can move).
    hit_turn_cap: bool = False
    # True when the episode was cancelled by a harbor agent timeout or carried a timeout exception.
    hit_timeout: bool = False

    @field_validator("attempt", mode="before")
    @classmethod
    def _reject_boolean_attempt(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("attempt must be an integer, not boolean")
        return value

    @field_validator("reward", mode="before")
    @classmethod
    def _reject_boolean_reward(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("reward must be numeric, not boolean")
        return value


class ScoreReport(BaseModel):
    """The scorecard for one exact candidate over one exact request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_hash: str = Field(strict=True, pattern=_DOC_HASH_PATTERN)
    request: ScoreRequest
    reward_mode: RewardMode
    cells: tuple[ScoreCell, ...]

    @field_validator("cells")
    @classmethod
    def _canonicalize_cells(cls, value: tuple[ScoreCell, ...]) -> tuple[ScoreCell, ...]:
        return tuple(sorted(value, key=lambda cell: (cell.task_id, cell.attempt)))

    @model_validator(mode="after")
    def _validate_matrix(self) -> ScoreReport:
        observed = [(cell.task_id, cell.attempt) for cell in self.cells]
        expected = {
            (task_id, attempt)
            for task_id in self.request.task_ids
            for attempt in range(1, self.request.attempts + 1)
        }
        duplicates = sorted({key for key in observed if observed.count(key) > 1})
        if duplicates:
            raise ValueError(f"duplicate score cell(s): {duplicates}")
        missing = sorted(expected - set(observed))
        extra = sorted(set(observed) - expected)
        if missing or extra:
            raise ValueError(f"score cells do not match request: missing={missing}, extra={extra}")
        return self

    @property
    def score(self) -> float:
        """The optimization objective: mean of per-task pass rates (equal task weight)."""
        by_task = self.by_task()
        return fmean(
            fmean(1.0 if cell.passed else 0.0 for cell in cells) for cells in by_task.values()
        )

    @property
    def pass_rate(self) -> float:
        """The flat fraction of passed cells, for reporting."""
        return fmean(1.0 if cell.passed else 0.0 for cell in self.cells)

    def by_task(self) -> dict[str, tuple[ScoreCell, ...]]:
        """Cells grouped by task in request order (attempts ascending within each task)."""
        return {
            task_id: tuple(cell for cell in self.cells if cell.task_id == task_id)
            for task_id in self.request.task_ids
        }

    def telemetry_summary(self) -> dict[str, JsonValue]:
        """Candidate-level token telemetry aggregated over cells that carry it.

        Deterministic and pure: it only reads the cells' populated telemetry fields (a cell with
        `turns is None` contributed no transcript and is skipped). Returns
        `{"n_with_telemetry": 0}` when no cell has telemetry, so a caller can always distinguish
        "no signal" from a real aggregate rather than guessing from an empty dict. This is the
        "impossible to miss" signal the proposer reads to decide which harness caps are binding:
        a high `turn_cap_hit_rate` means raising max_turns will help, while an `output_tokens_p90`
        far below the harness's max_output_tokens means that cap is not the bottleneck.
        """
        cells = [cell for cell in self.cells if cell.turns is not None]
        n = len(cells)
        if n == 0:
            return {"n_with_telemetry": 0}
        input_tokens = [cell.input_tokens for cell in cells if cell.input_tokens is not None]
        output_tokens = [cell.output_tokens for cell in cells if cell.output_tokens is not None]
        return {
            "n_with_telemetry": n,
            "turn_cap_hit_rate": fmean(1.0 if cell.hit_turn_cap else 0.0 for cell in cells),
            "timeout_rate": fmean(1.0 if cell.hit_timeout else 0.0 for cell in cells),
            "input_tokens_median": median(input_tokens) if input_tokens else None,
            "output_tokens_median": median(output_tokens) if output_tokens else None,
            "output_tokens_p90": _p90(output_tokens) if output_tokens else None,
        }


class Scorer(Protocol):
    """A synchronous candidate evaluator injected into a harness optimization loop.

    `request` declares the exact matrix every `score` call evaluates, so the loop can size
    budgets and validate reports without knowing the evaluator. `should_cancel` is polled at
    safe points; a scorer that observes it raises instead of returning a partial report.
    """

    @property
    def request(self) -> ScoreRequest: ...

    def score(
        self,
        doc: HarnessDoc,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> ScoreReport: ...
