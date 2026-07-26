"""Optional Weights & Biases tracking for distillation runs.

`build_tracker` turns the run config's `[wandb]` section into a
`DistillTracker`: the no-op `NullTracker` when tracking is disabled (the
default), or a `WandbTracker` streaming step metrics, eval solve rates,
sample rollout tables, and the final gate summary to a wandb run. The wandb
SDK stays an optional extra
(lazy import, mirroring the tinker SDK in `wmh.providers.tinker`), and
credentials are checked at init so a misconfigured run fails fast BEFORE any
paid rollout. After a successful init the contract inverts: a wandb failure
mid-run (network blip, service outage) logs one warning and every later
tracker call degrades to a no-op, because a dead dashboard must never abort a
paid training run. A restarted run continues its dashboard run: the wandb run
id persists in `<run_dir>/wandb-run.json` and later inits resume it (see
`WandbTracker` for the id/resume and step-monotonicity reasoning).
"""

from __future__ import annotations

import logging
import netrc
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from wmh.core.types import JsonObject, JsonValue
from wmh.distill.config import DistillConfig
from wmh.distill.offpolicy import OffPolicyMetrics
from wmh.distill.samples import SampleRollout
from wmh.distill.store import write_text_atomic

if TYPE_CHECKING:
    from wmh.distill.loop import StepMetrics, WarmupMetrics

logger = logging.getLogger(__name__)

WANDB_API_KEY_ENV = "WANDB_API_KEY"
WANDB_NETRC_MACHINE = "api.wandb.ai"

WANDB_RUN_FILE = "wandb-run.json"
"""Run-dir file persisting the wandb run id so a restart resumes the same run."""

_MISSING_WANDB_EXTRA = (
    "the wandb SDK is not installed; run `uv sync --extra distill` to enable "
    "[wandb] run tracking, or set wandb.enabled = false in the distill config"
)


class DistillTracker(Protocol):
    """The tracking slice the distillation loop emits to.

    Implementations must never raise from `log_step`, `log_warmup_step`,
    `log_eval`, `log_samples`, `log_summary`, or `finish` once constructed:
    the loop calls them inline with paid training work.
    """

    def log_step(self, step: int, metrics: StepMetrics) -> None:
        """Record one training step's metrics row."""
        ...

    def log_offpolicy_step(self, offpolicy_step: int, metrics: OffPolicyMetrics) -> None:
        """Record one off-policy step's metrics row (keys under `offpolicy/`)."""
        ...

    def log_warmup_step(self, warmup_step: int, metrics: WarmupMetrics) -> None:
        """Record one warmup step's metrics row (keys under `warmup/`)."""
        ...

    def log_eval(
        self,
        name: str,
        solve_rate: float,
        step: int | None,
        *,
        graded_solve_rate: float | None = None,
    ) -> None:
        """Record one eval batch's solve rate (None step means pre-training).

        `graded_solve_rate` is the graded test-pass companion, and None means the batch measured
        none (no readable test report, or a baseline imported from a run predating the metric):
        nothing is charted rather than a fabricated 0.0.
        """
        ...

    def log_samples(self, kind: str, step: int | None, samples: list[SampleRollout]) -> None:
        """Record one batch's rendered sample rollouts (None step means pre-training)."""
        ...

    def log_summary(
        self,
        *,
        gate_accepted: bool,
        gate_reason: str,
        teacher_solve_rate: float,
        student_before_solve_rate: float,
        student_after_solve_rate: float,
        total_usd: float,
        steps_completed: int,
    ) -> None:
        """Record the run's terminal outcome (the gate verdict and totals)."""
        ...

    def finish(self) -> None:
        """Flush and close the tracking run (idempotent best effort)."""
        ...


class NullTracker:
    """The disabled tracker: every call is a no-op."""

    def log_step(self, step: int, metrics: StepMetrics) -> None:
        """No-op."""

    def log_offpolicy_step(self, offpolicy_step: int, metrics: OffPolicyMetrics) -> None:
        """No-op."""

    def log_warmup_step(self, warmup_step: int, metrics: WarmupMetrics) -> None:
        """No-op."""

    def log_eval(
        self,
        name: str,
        solve_rate: float,
        step: int | None,
        *,
        graded_solve_rate: float | None = None,
    ) -> None:
        """No-op."""

    def log_samples(self, kind: str, step: int | None, samples: list[SampleRollout]) -> None:
        """No-op."""

    def log_summary(
        self,
        *,
        gate_accepted: bool,
        gate_reason: str,
        teacher_solve_rate: float,
        student_before_solve_rate: float,
        student_after_solve_rate: float,
        total_usd: float,
        steps_completed: int,
    ) -> None:
        """No-op."""

    def finish(self) -> None:
        """No-op."""


# -- the wandb SDK slice (typed so the lazy import stays checkable) -------------------------------


class WandbSummaryLike(Protocol):
    """The run-summary slice: `update` with plain JSON values."""

    def update(self, values: Mapping[str, JsonValue]) -> None:
        """Merge values into the run's summary."""
        ...


class WandbRunLike(Protocol):
    """The active-run slice: the id (persisted for resume) and the summary."""

    @property
    def id(self) -> str:
        """The run's wandb id (what a restart passes back to `init`)."""
        ...

    @property
    def summary(self) -> WandbSummaryLike:
        """The run's summary mapping."""
        ...


class WandbTableLike(Protocol):
    """An opaque wandb Table instance: constructed, logged, never read back."""


WandbLogValue = JsonValue | WandbTableLike
"""What one wandb.log payload value may be: a plain scalar or a table."""


class WandbModuleLike(Protocol):
    """The module-level wandb surface the tracker drives."""

    def init(
        self,
        *,
        project: str,
        entity: str | None,
        name: str,
        tags: list[str],
        # `dir` and `id` shadow builtins, but they are the wandb SDK's own
        # keyword names.
        dir: str,
        config: JsonObject,
        id: str | None,
        resume: Literal["allow"] | None,
    ) -> WandbRunLike:
        """Start (or, with an id and resume mode, continue) a wandb run."""
        ...

    # The SDK's own class name; structurally just a callable attribute.
    def Table(self, *, columns: list[str], data: list[list[JsonValue]]) -> WandbTableLike:
        """Build one wandb Table (a fresh one per `log_samples` call)."""
        ...

    def log(self, data: Mapping[str, WandbLogValue], *, step: int) -> None:
        """Log one row of metrics (or tables) at a step."""
        ...

    def finish(self) -> None:
        """Flush and close the active run."""
        ...


def _import_wandb() -> WandbModuleLike:
    """Lazily import the optional wandb SDK (the distill extra).

    Raises:
        ImportError: If the SDK is not installed; the message names the fix.
    """
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(_MISSING_WANDB_EXTRA) from exc
    return cast("WandbModuleLike", wandb)


def _netrc_has_wandb_login() -> bool:
    """Whether ~/.netrc carries an api.wandb.ai entry (a prior `wandb login`)."""
    try:
        entry = netrc.netrc().authenticators(WANDB_NETRC_MACHINE)
    except (FileNotFoundError, netrc.NetrcParseError):
        return False
    return entry is not None


def _require_wandb_credentials() -> None:
    """Fail fast when no wandb credentials exist (before any paid work).

    Raises:
        ValueError: When WANDB_API_KEY is unset AND no api.wandb.ai entry
            exists in ~/.netrc; the message names both fixes.
    """
    if os.environ.get(WANDB_API_KEY_ENV):
        return
    if _netrc_has_wandb_login():
        return
    raise ValueError(
        f"[wandb] tracking is enabled but no credentials were found: "
        f"{WANDB_API_KEY_ENV} is not set in the environment and ~/.netrc has no "
        f"{WANDB_NETRC_MACHINE} entry. Set {WANDB_API_KEY_ENV} to your API key, or "
        "run `wandb login` once to store it in ~/.netrc"
    )


class WandbRunRecord(BaseModel):
    """The `wandb-run.json` shape: the run id a restarted session resumes."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)


def _read_wandb_run_id(path: Path) -> str | None:
    """The persisted wandb run id, or None when absent or unreadable.

    A corrupt record is downgraded to a fresh run (with a warning) rather
    than raised: losing dashboard continuity must never block a paid run,
    and the caller rewrites the file after init either way.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        return WandbRunRecord.model_validate_json(text).run_id
    except ValidationError:
        logger.warning(
            "corrupt wandb run record at %s; starting a fresh wandb run and rewriting the file",
            path,
            exc_info=True,
        )
        return None


def _flatten_step_metrics(metrics: StepMetrics) -> dict[str, float | int | str]:
    """One step's metrics row as flat, namespaced wandb keys.

    Per-meter token counts land under `tokens/`, the step's priced spend
    under `cost/usd` (with the run's all-session total as `cost/usd_cum`),
    the per-stop-reason trial counts under `stop/<reason>`, and every other
    number under `train/`. The one non-numeric key kept is the objective
    (`train/loss`, e.g. `"ppo"`), because a chart of `train/advantage_mean`
    or `train/clip_fraction` means different things per mode and the row must
    say which one produced it. Other non-numeric fields (the sampler path)
    and unreported values (`reverse_kl_per_token`, `reward_mean`, the
    advantage stats, `pg_loss`, and `grad_norm` when None) are dropped:
    wandb charts numbers, and absent backend metrics are never fabricated.

    `train/scaffold_loss_rate` plus the `stop/` series is the pair that makes a
    harness-induced floor visible on the dashboard: the pi/Nemotron-3 runs sat
    at 88.8% scaffold loss with nothing to chart it against.

    `train/solve_rate` and `train/graded_solve_rate` are charted side by side:
    binary is the benchmark's own verdict and the gate's number, graded is the
    same trials at test resolution and the series with enough resolution to move
    on a 12-task batch. Read `train/graded_trials` beside it, since a graded rate
    over zero graded trials is a null measurement carrying 0.0.

    `train/entropy_per_token` and `train/mean_generation_tokens`, beside their
    `train/*_baseline` and `train/*_ratio` companions, are the degeneration
    pair: the ratio series is the one to chart, since only it is comparable
    across runs (each run measures its own baseline). A step that sampled
    nothing charts no entropy or length point rather than a fabricated zero.
    """
    explicit = {
        "student_prefill_tokens": "tokens/student_prefill",
        "student_cached_prefill_tokens": "tokens/student_cached_prefill",
        "student_sample_tokens": "tokens/student_sample",
        "student_train_tokens": "tokens/student_train",
        "teacher_prefill_tokens": "tokens/teacher_prefill",
        "teacher_cached_prefill_tokens": "tokens/teacher_cached_prefill",
        "teacher_sample_tokens": "tokens/teacher_sample",
        "usd": "cost/usd",
        "cumulative_usd": "cost/usd_cum",
    }
    payload: dict[str, float | int | str] = {}
    for key, value in metrics.model_dump(mode="json").items():
        if key == "loss" and isinstance(value, str):
            payload["train/loss"] = value
            continue
        if key == "stop_reason_counts":
            if isinstance(value, dict):
                payload.update(
                    {
                        f"stop/{reason}": count
                        for reason, count in value.items()
                        if isinstance(count, int)
                    }
                )
            continue
        if not isinstance(value, int | float) or isinstance(value, bool):
            continue
        payload[explicit.get(key, f"train/{key}")] = value
    return payload


class WandbTracker:
    """Streams a distillation run to Weights & Biases.

    Construction is strict (missing SDK or credentials raise, so a
    misconfigured run fails before spending anything); logging is forgiving
    (after a successful init, the first wandb failure logs one warning and
    every later call becomes a no-op, because a dead dashboard must never
    abort a paid training run).

    Restart continuity: the wandb run id is persisted to
    `<run_dir>/wandb-run.json` on first init, and when that file exists a
    later construction passes `id=<persisted>` with `resume="allow"` so a
    restarted run CONTINUES the same dashboard run ("allow" resumes the run
    when the id exists and starts one under that id otherwise). A missing or
    corrupt record starts fresh and rewrites the file. Resumed logging stays
    step-monotonic: `log_step` logs at the training step number, which only
    moves forward across sessions (a resume continues from the last
    checkpoint; the rare re-run of an un-checkpointed step re-logs at its own
    index, which wandb treats as an update to or drop at that step, never a
    decrease), and warmup rows always log at wandb step 0 under `warmup/`
    keys before any training row exists, so wandb never sees a step go
    backwards.

    Args:
        cfg: The validated run config; its `[wandb]` section names the run
            and its snapshot dump becomes the wandb run config.
        run_dir: The run's artifact directory; wandb files land under it.
        agent_name: The agent being distilled; names the run when
            `wandb.run_name` is unset.

    Raises:
        ImportError: If the wandb SDK is not installed (the distill extra).
        ValueError: If no credentials exist (see `_require_wandb_credentials`).
    """

    def __init__(self, cfg: DistillConfig, run_dir: Path, agent_name: str) -> None:
        wandb = _import_wandb()
        _require_wandb_credentials()
        run_name = cfg.wandb.run_name or f"{agent_name}-{run_dir.name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_record_path = run_dir / WANDB_RUN_FILE
        persisted_id = _read_wandb_run_id(run_record_path)
        self._wandb = wandb
        self._run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=run_name,
            tags=list(cfg.wandb.tags),
            dir=str(run_dir),
            # The same plain dict snapshot_toml renders, so the wandb run
            # config matches the run dir's config.toml exactly.
            config=cfg.model_dump(mode="json", exclude_none=True),
            id=persisted_id,
            resume="allow" if persisted_id is not None else None,
        )
        # Atomic like every other durable run-dir file: a crash between
        # wandb.init() and a torn write would leave the id absent or partial,
        # and the resumed session would silently start a SECOND dashboard run
        # instead of continuing this one.
        write_text_atomic(
            run_record_path, WandbRunRecord(run_id=self._run.id).model_dump_json(indent=2)
        )
        self._dead = False
        if persisted_id is not None:
            logger.info(
                "wandb tracking resumed: project %s, run %s (id %s)",
                cfg.wandb.project,
                run_name,
                persisted_id,
            )
        else:
            logger.info("wandb tracking started: project %s, run %s", cfg.wandb.project, run_name)

    def _guarded(self, action: Callable[[], None]) -> None:
        """Run one wandb call, degrading to a no-op if the dashboard dies.

        The first failure logs one warning and marks the tracker dead; every
        later call (including `finish`) is skipped silently. Training must
        keep going: the run's own artifacts (metrics.jsonl, eval reports) are
        unaffected by a lost dashboard.
        """
        if self._dead:
            return
        try:
            action()
        except Exception:  # noqa: BLE001 - any wandb failure degrades, never aborts the run
            self._dead = True
            logger.warning(
                "wandb logging failed; tracking is disabled for the rest of the run "
                "(training continues; the run dir keeps every artifact)",
                exc_info=True,
            )

    def log_step(self, step: int, metrics: StepMetrics) -> None:
        """Log one training step's flattened metrics row."""
        payload = _flatten_step_metrics(metrics)
        self._guarded(lambda: self._wandb.log(cast("Mapping[str, JsonValue]", payload), step=step))

    def log_offpolicy_step(self, offpolicy_step: int, metrics: OffPolicyMetrics) -> None:
        """Log one off-policy step's numeric fields under `offpolicy/` keys.

        Same wandb-step rule as the warmup rows below: the phase precedes
        training step 0 and wandb steps must never decrease, so every row logs
        at wandb step 0 with the phase-global index carried as
        `offpolicy/step`. The run dir's metrics.jsonl keeps every row.
        """
        self._log_phase_row("offpolicy", offpolicy_step, metrics)

    def log_warmup_step(self, warmup_step: int, metrics: WarmupMetrics) -> None:
        """Log one warmup step's numeric fields under `warmup/` keys.

        Warmup precedes training step 0 and wandb steps must never decrease,
        so every warmup row logs at wandb step 0 with the warmup index carried
        as `warmup/step` (later warmup rows overwrite same-step keys on the
        dashboard; the run dir's metrics.jsonl keeps every row).
        """
        self._log_phase_row("warmup", warmup_step, metrics)

    def _log_phase_row(self, prefix: str, index: int, metrics: BaseModel) -> None:
        """Log one pre-training phase row's numeric fields under `<prefix>/` keys."""
        payload: dict[str, float | int] = {f"{prefix}/step": index}
        for key, value in metrics.model_dump(mode="json").items():
            if not isinstance(value, int | float) or isinstance(value, bool):
                continue
            payload[f"{prefix}/{key}"] = value
        self._guarded(lambda: self._wandb.log(cast("Mapping[str, JsonValue]", payload), step=0))

    def log_eval(
        self,
        name: str,
        solve_rate: float,
        step: int | None,
        *,
        graded_solve_rate: float | None = None,
    ) -> None:
        """Log one eval batch's solve rate under `eval/<name>`, graded under `eval/<name>-graded`.

        The binary rate is always charted (it is the benchmark's own verdict and what the promotion
        gate reads); the graded companion is charted only when the batch measured one, so a batch
        with no readable test report leaves a gap in that series instead of a 0.0 point.
        """
        at_step = step if step is not None else 0
        payload: dict[str, JsonValue] = {f"eval/{name}": solve_rate}
        if graded_solve_rate is not None:
            payload[f"eval/{name}-graded"] = graded_solve_rate
        self._guarded(lambda: self._wandb.log(payload, step=at_step))

    def log_samples(self, kind: str, step: int | None, samples: list[SampleRollout]) -> None:
        """Log one batch's rendered rollouts as a fresh wandb Table per call.

        A wandb Table is immutable once logged (current SDKs reject
        re-logging a mutated Table object, and incremental appends need
        artifact round-trips), so the tracker keeps it simple: each call
        builds a fresh table with columns kind/step/trial/reward/text and
        logs it under a step-qualified key, `samples/<kind>-<step>`
        (`samples/<kind>` when step is None), so every batch's sample set
        stays addressable on the dashboard. Pre-training batches (step None)
        chart at wandb step 0, matching `log_eval`.
        """
        if not samples:
            return
        at_step = step if step is not None else 0
        key = f"samples/{kind}" if step is None else f"samples/{kind}-{step:04d}"

        def _log() -> None:
            table = self._wandb.Table(
                columns=["kind", "step", "trial", "reward", "text"],
                data=[
                    [kind, step, sample.trial_name, sample.reward, sample.text]
                    for sample in samples
                ],
            )
            self._wandb.log({key: table}, step=at_step)

        self._guarded(_log)

    def log_summary(
        self,
        *,
        gate_accepted: bool,
        gate_reason: str,
        teacher_solve_rate: float,
        student_before_solve_rate: float,
        student_after_solve_rate: float,
        total_usd: float,
        steps_completed: int,
    ) -> None:
        """Record the run's terminal outcome in the wandb run summary."""
        values: dict[str, JsonValue] = {
            "gate_accepted": gate_accepted,
            "gate_reason": gate_reason,
            "teacher_solve_rate": teacher_solve_rate,
            "student_before_solve_rate": student_before_solve_rate,
            "student_after_solve_rate": student_after_solve_rate,
            "total_usd": total_usd,
            "steps_completed": steps_completed,
        }
        self._guarded(lambda: self._run.summary.update(values))

    def finish(self) -> None:
        """Flush and close the wandb run (skipped once the tracker is dead)."""
        self._guarded(self._wandb.finish)


def build_tracker(cfg: DistillConfig, run_dir: Path, agent_name: str) -> DistillTracker:
    """The tracker for one run: `WandbTracker` when enabled, else `NullTracker`.

    Args:
        cfg: The validated run config (`cfg.wandb.enabled` decides).
        run_dir: The run's artifact directory.
        agent_name: The agent being distilled (names the default wandb run).

    Returns:
        The tracker the loop should emit to.

    Raises:
        ImportError: Tracking enabled but the wandb SDK is not installed.
        ValueError: Tracking enabled but no wandb credentials exist.
    """
    if not cfg.wandb.enabled:
        return NullTracker()
    return WandbTracker(cfg, run_dir, agent_name)
