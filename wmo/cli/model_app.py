"""`wmo optimize distill`: train the agent MODEL, leaving its harness pinned.

The third member of the optimizer family, beside `wmo optimize harness`
(prompt surfaces) and `wmo optimize route` (routing policy). Where those
produce a `prompt` or a `routing_policy` artifact, this one produces an
`adapter`: `run` drives one distillation of a Tinker LoRA student from real
benchmark rollouts (the config selects the source: harbor's own terminus-2
agent, or tau2-bench's own harness), and `report` reads a finished run dir
back.

`probe` comes BEFORE either of those and costs nothing: it reads the routing
sweep's outcome matrix and answers whether this workload has a teacher gap at
all (`wmo.optimize.routing.teacher`). Most workloads do not, and the cheapest run is
the one the evidence says to skip.

`run` owns the run's CLI lifecycle: load and pin the inputs (config, task
splits, the harness document supplying the rollout params), project the run
cost into a confirmation table, drive `run_distillation` with progress
rendering, and print the gate verdict plus the serving handoff. The optional
`--promote` step writes `[models.agent]` through the settings save path after
an explicit confirmation.

Run-dir pinning mirrors `run-config.json` in the harbor search flow: a fresh
run records its CLI-level inputs in `distill-run.json` (task splits, backend,
the exact harness version and doc hash), and a resume reuses that record
instead of live flags, rejecting explicit flags that conflict with it. The run
config itself is snapshotted by the run store as `config.toml`, which is what
a bare `--resume` (no `--config`) loads.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import typer
from pydantic import BaseModel, ConfigDict, ValidationError
from rich.console import Console
from rich.markup import escape
from rich.prompt import Confirm
from rich.table import Table

import wmo.cli.model_app as _self
from wmo.cli.consent import can_prompt, require_spend_consent
from wmo.common.config import ARTIFACT_DIR

if TYPE_CHECKING:
    # Type-only: real imports are local to the commands and helpers that construct or inspect
    # these values, so importing this module never pulls the distill/harness/optimize bodies
    # behind it.
    from wmo.common.core.types import JsonObject
    from wmo.optimize.model.config import DistillConfig
    from wmo.optimize.model.cost import CostEstimate
    from wmo.optimize.model.gate import DistillGateRecord
    from wmo.optimize.model.loop import DistillEvalReport, DistillProgress, DistillResult
    from wmo.optimize.model.store import AdapterStore, DistillRunStore
    from wmo.optimize.routing.teacher import TeacherSearchVerdict
    from wmo.runtime.harness.doc import HarnessDoc
    from wmo.runtime.harness.e2b_reap import CapacityCheck

# Literal mirrors of constants that otherwise live behind a heavy import (`wmo.optimize.model.loop`,
# `wmo.optimize.routing.teacher`). Typer evaluates Option defaults at command-definition time,
# so these have to be values, not names imported from those modules; the real constants are
# re-imported inside the command bodies that need their behavior.
_DEFAULT_DISTILL_HARNESS = "pi"
_DEFAULT_MIN_GAP = 0.10

DISTILL_RUN_RECORD = "distill-run.json"
"""The CLI-level pin file inside the run dir (see `DistillCliRunRecord`)."""

PROBE_EXIT_NO_GAP = 3
"""`probe` exit code: the matrix shows no teacher gap, so training is refused."""

PROBE_EXIT_INSUFFICIENT = 4
"""`probe` exit code: the matrix is too thin to decide either way.

Distinct from `PROBE_EXIT_NO_GAP` because the two call for opposite actions (ship the cheap model
against sweep more scenarios), and distinct from 1 and 2 so a script can tell a verdict from a
crash or a usage error.
"""

_PI_NODE_RUNTIME = "pi-node"

model_app = typer.Typer(
    help="Train the agent model itself: distillation of a Tinker LoRA student from real "
    "benchmark rollouts (harbor or tau2, config-selected), gated on held-out solve rates.",
    no_args_is_help=True,
)

_console = Console()


def _load_harbor_task_ids(path: Path) -> tuple[str, ...]:
    """Load the exact ordered task-id list, validated by the canonical score request rules."""
    from wmo.runtime.harness.scoring import ScoreRequest

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise typer.BadParameter(f"cannot load task ids from {path}: {error}") from error
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise typer.BadParameter(f"{path} must contain one JSON array of task-id strings")
    try:
        request = ScoreRequest(task_ids=tuple(raw), attempts=1)
    except ValidationError as error:
        raise typer.BadParameter(f"invalid task ids in {path}: {error}") from error
    return request.task_ids


@model_app.command("run")
def run(
    ctx: typer.Context,
    config: str = typer.Option(
        None,
        "--config",
        help="The run TOML (student, teacher, EXACTLY ONE of harbor/tau2, plus rollout, "
        "train, sampling, warmup, eval, gate, pricing, budget, tripwire, wandb). Required "
        "to start a run; "
        "a resume reuses the run dir's config.toml snapshot, and passing it on a resume is "
        "how you raise budget.max_usd.",
    ),
    run_dir: str = typer.Option(
        ...,
        "--run-dir",
        help="Directory holding ALL durable run state (config snapshot, metrics, "
        "checkpoints, evals, rollout artifacts). Always required.",
    ),
    task_ids: str = typer.Option(
        None,
        "--task-ids",
        help="JSON file with the exact train task-id list; rollouts and interim evals run "
        "here. Required to start a run.",
    ),
    holdout_task_ids: str = typer.Option(
        None,
        "--holdout-task-ids",
        help="JSON file with the exact holdout task-id list; the baselines and the promotion "
        "gate are measured here, disjoint from --task-ids. Required to start a run.",
    ),
    harness: str = typer.Option(
        _DEFAULT_DISTILL_HARNESS,
        "--harness",
        help="Stored harness document supplying the rollout params (temperature, max turns, "
        "max output tokens); under the harbor source its hash also keys every harbor job "
        "(the harbor agent is always terminus-2, never this document's runtime), and the "
        f"tau2 source takes its params from the config alone. The bare literal "
        f"{_DEFAULT_DISTILL_HARNESS!r} is the built-in default agent; 'name@ref' pins a stored "
        "version. Pinned for the whole run.",
    ),
    backend: str = typer.Option(
        None,
        "--backend",
        help="Override the rollout source's backend (harbor.backend or tau2.backend): local "
        "(runs on this machine) or e2b (E2B sandboxes; harbor only today, needs E2B_API_KEY).",
    ),
    resume: bool = typer.Option(False, "--resume", help="Continue the run recorded in --run-dir."),
    promote: bool = typer.Option(
        False,
        "--promote",
        help="After an accepted gate, offer to point the models.agent role in settings.toml "
        "at the distilled adapter (always asks for confirmation).",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Consent to the projected spend up front. Required in a non-interactive "
        "session (CI, cron, piped output, redirected input), where the run otherwise "
        "refuses to start.",
    ),
    root: str = typer.Option(ARTIFACT_DIR, "--root", help="Project dir."),
) -> None:
    """Train (or resume) an agent model by distillation on real benchmark tasks.

        wmo optimize distill run --config run.toml --run-dir runs/d1 \\
          --task-ids train.json --holdout-task-ids holdout.json --backend e2b --yes

    The config-selected rollout source (harbor's terminus-2 agent, or real
    tau2-bench episodes through the loopback proxy) rolls out while sampling
    the student's current Tinker LoRA weights, a larger teacher scores or
    demonstrates, and training nudges the student toward the teacher (warmup
    cross-entropy, then optional per-token reverse-KL OPD steps). A held-out
    gate compares teacher, student-before, and student-after solve rates, and
    only an adapter that closes enough of the gap is promoted.

    The harness is NOT the subject here and is never edited: it is pinned for
    the whole run. Search the scaffold with `wmo optimize harness` instead.
    """
    run_distill(
        _console,
        harness_name=harness,
        harness_explicit=_explicit(ctx, "harness"),
        config_path=config,
        task_ids_path=task_ids,
        holdout_task_ids_path=holdout_task_ids,
        run_dir=run_dir,
        backend=backend,
        resume=resume,
        yes=yes,
        promote=promote,
        root=root,
    )


@model_app.command("report")
def report(
    run_dir: str = typer.Option(
        ..., "--run-dir", help="A finished (or aborted) run directory to read back."
    ),
) -> None:
    """Print a run's gate verdict and its held-out before/after table.

    Reads only what the run dir already persisted (`gate.json`, `evals/*.json`,
    `metrics.jsonl`), so it is free to run and safe on a live run dir:

        wmo optimize distill report --run-dir runs/d1
    """
    from wmo.optimize.model.store import DistillRunStore

    store = DistillRunStore(run_dir)
    gate = _load_gate(store)
    color = "green" if gate.accepted else "yellow"
    _console.print(f"[{color}]gate[/{color}] {escape(gate.reason)}")
    _console.print(_solve_rate_table(store, gate))
    _print_trained_artifact(_console, store)
    _print_paired_delta(_console, store, gate)
    _print_training_summary(_console, store)


@model_app.command("probe")
def probe(
    matrix_file: str = typer.Argument(
        ...,
        help="The outcome matrix to read: `<model>/optimize/matrix.json`, or wherever "
        "`wmo optimize route sweep --out` put one.",
    ),
    student: str = typer.Option(
        None,
        "--student",
        help="Pool model the distillation would train. Default: the cheapest measured model, "
        "which is the one whose price makes distillation worth doing at all.",
    ),
    min_gap: float = typer.Option(
        _DEFAULT_MIN_GAP,
        "--min-gap",
        min=0.0,
        max=1.0,
        help="Reward points (as a fraction of 1.0) a teacher must beat the student by. The "
        "default 0.10 is 10 points.",
    ),
) -> None:
    """Ask a measured matrix whether this workload has a teacher worth distilling from.

    Free and read-only: the sweep already bought this evidence, so the gate is arithmetic over
    the file.

        wmo optimize distill probe .wmo/models/tau-bench/optimize/matrix.json

    It prints every candidate's paired gain over the student, with a 95% interval and the price
    it would teach at, then one verdict line to act on. Exit codes, so a script can branch
    without parsing the text: 0 = distill (the gap is real; the named teacher is the cheapest
    sufficient one), 3 = do not distill (no gap, so training has nothing to teach), 4 =
    insufficient evidence (this matrix is too thin to say either way; sweep more scenarios).
    Anything else is the usual usage or IO failure.
    """
    from wmo.optimize.routing.outcomes import OutcomeMatrix
    from wmo.optimize.routing.teacher import select_teacher

    path = Path(matrix_file)
    if not path.is_file():
        raise typer.BadParameter(
            f"no outcome matrix at {path}; a sweep writes one to "
            f"`<model>/optimize/matrix.json`, so run `wmo optimize model <world-model>` (or "
            f"`wmo optimize route sweep`) before probing it"
        )
    try:
        matrix = OutcomeMatrix.load(path)
    except ValidationError as exc:
        raise typer.BadParameter(f"{path} is not a readable outcome matrix: {exc}") from exc
    try:
        verdict = select_teacher(matrix, student=student, min_gap=min_gap)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    _console.print(_teacher_gain_table(verdict, path))
    color = "green" if verdict.should_distill else "yellow"
    _console.print(f"[{color}]{verdict.decision}[/{color}] {escape(verdict.reason)}")
    if verdict.unmeasured_models:
        _console.print(
            f"[dim]not compared (no scenario scored alongside '{escape(verdict.student)}'): "
            f"{escape(', '.join(verdict.unmeasured_models))}[/dim]"
        )
    if verdict.decision == "do_not_distill":
        raise typer.Exit(code=PROBE_EXIT_NO_GAP)
    if verdict.decision == "insufficient_evidence":
        raise typer.Exit(code=PROBE_EXIT_INSUFFICIENT)


def _teacher_gain_table(verdict: TeacherSearchVerdict, matrix_file: Path) -> Table:
    """Every candidate's paired gain over the student, best first.

    Six columns, not nine: a gain and its interval belong in one cell (they are one measurement),
    and the student's own reward is the same on every row, so it goes in the title. At an
    80-column terminal the wider layout truncated every cell to an ellipsis, which is worse than
    omitting the columns outright.

    The price column carries its unit in its header, because the ladder is measured dollars per
    completed task when the matrix recorded spend and list dollars per Mtok when it did not, and
    a bare number would be read as the wrong one.
    """
    unit = "$/task" if verdict.price_basis == "measured" else "$/Mtok"
    measured = verdict.price_basis == "measured"
    unit_note = (
        "this matrix's own cost per completed task"
        if measured
        else "list price per 1M tokens, input + output, which is the fallback whenever some "
        "model has no measured cost per completed task (an unpriced entry, or a model that "
        "completed nothing)"
    )
    table = Table(
        title=f"Teacher search against '{verdict.student}' "
        f"({verdict.n_scenarios} scored scenarios, {matrix_file})",
        caption=f"n = scenarios shared with the student; gains are paired over those. "
        f"{unit} is {unit_note}. Bar: {verdict.min_gap * 100:.1f} points over "
        f"{verdict.min_scenarios}+ shared scenarios, interval excluding zero. "
        f"A teacher (*) keeps {verdict.sufficiency * 100:.0f}% of the best gain.",
        caption_justify="left",
    )
    table.add_column("Model", overflow="fold")
    table.add_column("Tier")
    table.add_column("n", justify="right")
    table.add_column("Reward", justify="right")
    table.add_column("Gain in points (95% CI)", justify="right")
    table.add_column(unit, justify="right")
    table.add_column("Gap?")
    for row in verdict.gains:
        interval = (
            f"({row.ci_low * 100:+.1f} to {row.ci_high * 100:+.1f})"
            if row.ci_low is not None and row.ci_high is not None
            else "(no interval: one scenario)"
        )
        table.add_row(
            f"{row.model} *" if row.model == verdict.teacher else row.model,
            row.tier,
            str(row.n_scenarios),
            f"{row.mean_reward:.3f}",
            f"{row.mean_gain * 100:+.1f} {interval}",
            _probe_price(row.price, measured=measured),
            "yes" if row.clears_gap else "no",
        )
    return table


def _probe_price(price: float | None, *, measured: bool) -> str:
    """A ladder key at the precision its unit deserves: fractions of a cent, or dollars a Mtok."""
    if price is None:
        return "unpriced"
    return f"{price:.4f}" if measured else f"{price:.2f}"


def _explicit(ctx: typer.Context, param: str) -> bool:
    """Whether `param` was explicitly passed on the command line.

    Compared by enum NAME: typer vendors click, so its ParameterSource enum is not
    click.core's class and an identity check would silently never match.
    """
    source = ctx.get_parameter_source(param)
    return source is not None and source.name == "COMMANDLINE"


class DistillCliRunRecord(BaseModel):
    """The CLI inputs pinned into `distill-run.json` when a distill run starts.

    A resume command carries only `--run-dir` (that is what a budget abort
    prints), so everything else the CLI resolved at start is recorded here and
    reloaded on resume; explicit flags that conflict with the record are
    rejected instead of silently changing what is being trained or gated.
    """

    model_config = ConfigDict(frozen=True)

    agent: str
    """The `--harness` value exactly as given (may carry an @ref).

    Named `agent` because that is the key already written into run dirs; it is
    the harness document supplying the rollout params, never the executing
    agent (harbor always runs terminus-2).
    """

    backend: Literal["local", "e2b"]
    seed_version: int | None
    """The stored harness version; None means the built-in default agent."""

    seed_doc_hash: str
    """The resolved harness document's hash; a resume must re-resolve to it."""

    train_task_ids: tuple[str, ...]
    holdout_task_ids: tuple[str, ...]


def run_distill(
    console: Console,
    *,
    harness_name: str,
    harness_explicit: bool,
    config_path: str | None,
    task_ids_path: str | None,
    holdout_task_ids_path: str | None,
    run_dir: str,
    backend: str | None,
    resume: bool,
    yes: bool,
    promote: bool,
    root: str,
) -> None:
    """Run (or resume) one on-policy distillation from the CLI.

    Args:
        console: The CLI's rich console (product output goes through it).
        harness_name: The `--harness` value; the bare default literal is the
            built-in default agent, 'name@ref' loads a stored version.
        harness_explicit: Whether `--harness` was typed rather than defaulted.
            A resume that did not type it adopts the recorded value instead of
            conflicting with it, which is why the printed resume command may
            omit the flag.
        config_path: The per-run TOML; required to start a fresh run. On
            resume None loads the run dir's config.toml snapshot, and an
            explicit path wins over it (the documented budget-abort recovery
            is editing budget.max_usd and resuming).
        task_ids_path: JSON array of train task ids; required to start.
        holdout_task_ids_path: JSON array of holdout task ids; required to
            start. Baselines and the gate are measured here.
        run_dir: The run's durable state directory; always required.
        backend: An explicit `--backend` override for the rollout source's
            backend (harbor or tau2), or None when the flag was not given.
        resume: Continue the run recorded in `run_dir`.
        yes: Consent to the projected spend up front, and required on a
            non-interactive session where there is nobody to ask (see
            `_confirm_cost` for the one case where confirmation is forced
            anyway).
        promote: After an accepted gate, offer to write `[models.agent]`
            pointing at the distilled adapter (explicit confirmation).
        root: The project dir (harness store, adapter store, settings).

    Raises:
        typer.BadParameter: On any invalid or conflicting input; the message
            names the flag and what to do.
        typer.Exit: When the user declines a confirmation (code 0) or the
            run fails/aborts (code 1).
    """
    # Deferred import: harness_app registers this module's typer app at module
    # scope, so importing its helpers back at module scope would be a circular
    # import.
    from wmo.optimize.model.cost import estimate_run_cost
    from wmo.optimize.model.loop import (
        DEFAULT_DISTILL_HARNESS,
        DistillBudgetError,
    )
    from wmo.optimize.model.store import AdapterStore, DistillRunStore
    from wmo.runtime.harness.store import write_json_atomic

    backend_override: Literal["local", "e2b"] | None
    if backend is None:
        backend_override = None
    elif backend == "e2b":
        backend_override = "e2b"
    elif backend == "local":
        backend_override = "local"
    else:
        raise typer.BadParameter(f"unknown --backend {backend!r}; choose local or e2b")

    run_path = Path(run_dir)
    record_path = run_path / DISTILL_RUN_RECORD
    store = DistillRunStore(run_path)
    seed_version: int | None
    if resume:
        record = _load_record(record_path)
        _reject_resume_conflicts(
            record,
            # A resume that did not type --harness adopts the record rather than
            # conflicting with the option's default, which is what lets the
            # printed resume command omit the flag at the default value.
            harness_name=harness_name if harness_explicit else None,
            backend=backend_override,
            task_ids_path=task_ids_path,
            holdout_task_ids_path=holdout_task_ids_path,
            load_task_ids=_load_harbor_task_ids,
        )
        harness_name = record.agent
        train_ids = record.train_task_ids
        holdout_ids = record.holdout_task_ids
        cfg = _load_config(Path(config_path) if config_path is not None else store.config_path)
        base, seed_doc = _pinned_seed_doc(root, record)
        seed_version = record.seed_version
        effective_backend = record.backend
    else:
        if store.config_path.exists():
            raise typer.BadParameter(
                f"{run_path} already holds a distillation run; pass --resume to "
                "continue it or choose a fresh --run-dir"
            )
        if record_path.exists():
            # The record is written before the loop starts, but the loop's very
            # first durable action is the config.toml snapshot: a record with no
            # snapshot means a previous start failed before doing (or spending)
            # anything, so treat the dir as fresh instead of bricking it.
            console.print(
                f"[yellow]note[/yellow] {run_path} holds a run record from a start "
                "that never began (no config.toml snapshot); starting fresh"
            )
        missing = [
            flag
            for flag, value in (
                ("--config", config_path),
                ("--task-ids", task_ids_path),
                ("--holdout-task-ids", holdout_task_ids_path),
            )
            if value is None
        ]
        if missing:
            raise typer.BadParameter(
                f"{', '.join(missing)} required to start a distillation run "
                "(a resume reuses the run dir's recorded inputs instead)"
            )
        assert config_path is not None  # narrowed by the missing check
        assert task_ids_path is not None and holdout_task_ids_path is not None
        cfg = _load_config(Path(config_path))
        train_ids = _load_harbor_task_ids(Path(task_ids_path))
        holdout_ids = _load_harbor_task_ids(Path(holdout_task_ids_path))
        base, seed_doc, seed_version = _resolve_seed_doc(root, harness_name)
        effective_backend = (
            backend_override if backend_override is not None else _source_backend(cfg)
        )

    overlap = sorted(set(train_ids) & set(holdout_ids))
    if overlap:
        raise typer.BadParameter(
            f"task id(s) {', '.join(overlap)} appear in BOTH --task-ids and "
            "--holdout-task-ids; the gate is only meaningful on tasks the student "
            "never trained on, so make the splits disjoint"
        )
    if effective_backend != _source_backend(cfg):
        source = cfg.rollout_source
        section = cfg.harbor if source == "harbor" else cfg.tau2
        assert section is not None  # exactly-one source, validated by the config
        cfg = cfg.model_copy(
            update={source: section.model_copy(update={"backend": effective_backend})}
        )
    runtime_kind = seed_doc.runtime_kind()
    if runtime_kind != _PI_NODE_RUNTIME:
        raise typer.BadParameter(
            f"distillation rollouts read their params from a pi-node harness document, "
            f"but --harness {harness_name!r} has runtime kind {runtime_kind!r}; pass a "
            f"pi-node harness (the built-in {DEFAULT_DISTILL_HARNESS!r} agent, or a "
            "version optimized from it)"
        )
    if cfg.harbor is not None:
        template_path = Path(cfg.harbor.job_template)
        if not template_path.is_file():
            raise typer.BadParameter(
                f"harbor.job_template {template_path} does not exist; point the distill "
                "config's [harbor] job_template at the harbor JobConfig YAML/JSON the "
                "rollouts should run"
            )
        if effective_backend == "e2b":
            _preflight_e2b_capacity(console, trial_concurrency=cfg.train.trial_concurrency)
    else:
        _preflight_tau2(cfg, [*train_ids, *holdout_ids])

    console.print(
        f"distilling [bold]{base}[/bold]: student {cfg.student.base_model} <- teacher "
        f"{cfg.teacher.checkpoint or cfg.teacher.model}, {cfg.train.steps} step(s) x "
        f"{cfg.train.tasks_per_batch} task(s) x {cfg.train.group_size} attempt(s), "
        f"{len(train_ids)} train / {len(holdout_ids)} holdout task(s), "
        f"backend {effective_backend} -> {run_path}"
    )
    estimate = estimate_run_cost(cfg, len(train_ids), len(holdout_ids))
    _print_cost_estimate(console, cfg, estimate)
    _confirm_cost(console, estimate, cfg.budget.max_usd, yes=yes)

    if not resume:
        # Recorded only now: inputs validated and the user confirmed, so a
        # declined or failed start never poisons the run dir.
        record = DistillCliRunRecord(
            agent=harness_name,
            backend=effective_backend,
            seed_version=seed_version,
            seed_doc_hash=seed_doc.doc_hash,
            train_task_ids=train_ids,
            holdout_task_ids=holdout_ids,
        )
        run_path.mkdir(parents=True, exist_ok=True)
        write_json_atomic(record_path, record.model_dump(mode="json"))

    def _on_progress(event: DistillProgress) -> None:
        spend = f" (${event.spent_usd:.2f} spent)" if event.spent_usd > 0 else ""
        # The literal bracket is escaped so rich does not eat the phase as a markup tag.
        console.print(f"  \\[{event.phase}] {escape(event.message)}{spend}")

    try:
        run_distill_fn = cast(Any, _self.run_distillation)
        result = run_distill_fn(
            base,
            cfg,
            seed_doc,
            list(train_ids),
            list(holdout_ids),
            run_path,
            resume=resume,
            on_progress=_on_progress,
            adapter_store=AdapterStore(root),
            # Resume commands must print the --harness string as typed (it may
            # carry an @ref that `base` strips), or the printed command would
            # trip the CLI's resume conflict check.
            cli_agent=harness_name,
        )
    except DistillBudgetError as exc:
        console.print(f"[red]budget exhausted[/red] {escape(str(exc))}")
        console.print(f"resume with: [bold]{escape(exc.resume_command)}[/bold]", soft_wrap=True)
        raise typer.Exit(1) from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except (RuntimeError, ImportError) as exc:
        console.print(f"[red]distillation failed[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    _print_result(
        console, result, store, adapters=AdapterStore(root), base_model=cfg.student.base_model
    )
    if promote:
        _maybe_promote(console, result, cfg, root)


# -- input resolution ------------------------------------------------------------------------


def _source_backend(cfg: DistillConfig) -> Literal["local", "e2b"]:
    """The configured rollout source's backend (the `--backend` default)."""
    if cfg.harbor is not None:
        return cfg.harbor.backend
    assert cfg.tau2 is not None  # exactly-one source, validated by the config
    return cfg.tau2.backend


def _preflight_tau2(cfg: DistillConfig, task_ids: Sequence[str]) -> None:
    """Fail a tau2-source run before anything is spent, naming the fix.

    Checks the pieces a tau2 rollout cannot run without: a wired backend, the
    tau2 CLI in its own venv, the data directory, well-formed composite task
    ids in both splits, and (for an azure/ user simulator) the litellm Azure
    credentials, by NAME only.

    Args:
        cfg: The validated run config; `cfg.tau2` must be set.
        task_ids: Every task id the run will draw (train plus holdout).

    Raises:
        typer.BadParameter: If any prerequisite is missing.
    """
    from wmo.optimize.model.tau2 import parse_tau2_task_id

    assert cfg.tau2 is not None
    if cfg.tau2.backend == "e2b":
        raise typer.BadParameter(
            "tau2.backend = 'e2b' is not wired yet; run with backend 'local' (the tau2 "
            "runner is an API-only subprocess, so the heavy work is remote either way)"
        )
    tau2_bin = Path(cfg.tau2.tau2_bin)
    if not tau2_bin.is_file():
        # Self-contained on purpose: tau2-bench is an external clone no wheel can ship, so a
        # pip-installed user has no repo file to be pointed at. The one-time setup is three
        # commands, so the message carries them instead of a path that may not exist.
        raise typer.BadParameter(
            f"tau2.tau2_bin {tau2_bin} does not exist; tau2-bench runs from its own Python 3.13 "
            "venv, so set it up once (`git clone --depth 1 "
            "https://github.com/sierra-research/tau2-bench && uv venv --python 3.13 .venv && "
            "uv pip install --python .venv ./tau2-bench audioop-lts boto3`) and point "
            "tau2.tau2_bin at that venv's CLI, <where-you-ran-it>/.venv/bin/tau2"
        )
    data_dir = Path(cfg.tau2.data_dir)
    if not data_dir.is_dir():
        raise typer.BadParameter(
            f"tau2.data_dir {data_dir} does not exist; point it at the tau2-bench clone's "
            "data directory (exported to every runner as TAU2_DATA_DIR)"
        )
    for task_id in task_ids:
        try:
            parse_tau2_task_id(task_id)
        except ValueError as error:
            raise typer.BadParameter(str(error)) from error
    if cfg.tau2.user_llm.startswith("azure/"):
        missing = [
            name
            for name in ("AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION")
            if not os.environ.get(name)
        ]
        if missing:
            raise typer.BadParameter(
                f"tau2.user_llm {cfg.tau2.user_llm!r} runs the user simulator through "
                f"litellm's azure/ route, which needs {', '.join(missing)} in the "
                "environment (set them in the gitignored .env; never commit values)"
            )


def _load_config(path: Path) -> DistillConfig:
    """Load the run TOML, turning load failures into usage errors."""
    from wmo.optimize.model.config import load_distill_config

    try:
        return load_distill_config(path)
    except FileNotFoundError as exc:
        raise typer.BadParameter(
            f"{exc} (a fresh run needs --config; a resume reads the run dir's config.toml snapshot)"
        ) from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except ImportError as exc:
        # Some sections can only be VALIDATED with the distill extra installed:
        # `[rollout.renderers]` resolves its names through tinker-cookbook. pydantic re-raises a
        # non-ValueError out of a field validator untouched, so without this a missing extra
        # (the state every shipped reference config lands in on a plain `pip install`) escapes
        # as a traceback instead of the install command the user needs.
        raise typer.BadParameter(f"cannot load {path}: {exc}") from exc


def _load_record(path: Path) -> DistillCliRunRecord:
    """Load the pinned `distill-run.json` for a resume."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise typer.BadParameter(
            f"--resume found no {DISTILL_RUN_RECORD} under {path.parent}; start the "
            "run once without --resume"
        ) from exc
    try:
        return DistillCliRunRecord.model_validate_json(text)
    except ValidationError as exc:
        raise typer.BadParameter(f"cannot load {path}: {exc}") from exc


def _reject_resume_conflicts(
    record: DistillCliRunRecord,
    *,
    harness_name: str | None,
    backend: str | None,
    task_ids_path: str | None,
    holdout_task_ids_path: str | None,
    load_task_ids: Callable[[Path], tuple[str, ...]],
) -> None:
    """Reject explicit flags that conflict with the recorded run inputs.

    Every flag is compared only when it was actually typed (None means it was
    not), so a resume that carries just `--run-dir --resume` adopts the record
    wholesale instead of colliding with an option default.
    """
    conflicts: list[str] = []
    if harness_name is not None and harness_name != record.agent:
        conflicts.append(f"--harness {harness_name!r} != recorded {record.agent!r}")
    if backend is not None and backend != record.backend:
        conflicts.append(f"--backend {backend!r} != recorded {record.backend!r}")
    if task_ids_path is not None and load_task_ids(Path(task_ids_path)) != record.train_task_ids:
        conflicts.append("--task-ids differs from the recorded train split")
    if (
        holdout_task_ids_path is not None
        and load_task_ids(Path(holdout_task_ids_path)) != record.holdout_task_ids
    ):
        conflicts.append("--holdout-task-ids differs from the recorded holdout split")
    if conflicts:
        raise typer.BadParameter(
            f"--resume uses the recorded {DISTILL_RUN_RECORD}; conflicting flag(s): "
            + "; ".join(conflicts)
            + ". Drop them to continue this run, or start a fresh --run-dir"
        )


def _resolve_seed_doc(root: str, harness_ref: str) -> tuple[str, HarnessDoc, int | None]:
    """Resolve `--harness` to the document the trials read their params from.

    Mirrors the harbor search's seed protocol: the bare default literal is
    ALWAYS the built-in agent; 'name@ref' loads a stored version. Returns the
    base name, the document, and the resolved store version (None for the
    built-in seed) so a resume can pin exactly what the run started from.
    """
    from wmo.common.config.store import validate_name
    from wmo.optimize.model.loop import DEFAULT_DISTILL_HARNESS
    from wmo.runtime.agents.default import default_agent
    from wmo.runtime.harness.store import HarnessStore

    base, _, ref = harness_ref.partition("@")
    try:
        validate_name(base)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if base == DEFAULT_DISTILL_HARNESS and not ref:
        return base, default_agent(base), None
    try:
        doc = HarnessStore(root).load(base, ref or None)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(
            f"{exc}; the built-in default agent is the literal {DEFAULT_DISTILL_HARNESS!r}"
        ) from exc
    return base, doc, doc.version


def _pinned_seed_doc(root: str, record: DistillCliRunRecord) -> tuple[str, HarnessDoc]:
    """Re-resolve the recorded seed for a resume, never a live movable ref.

    The record pins the exact version and doc hash, so champion movement (or
    any store edit) between sessions cannot silently change which harness the
    remaining trials run.
    """
    from wmo.runtime.agents.default import default_agent
    from wmo.runtime.harness.store import HarnessStore

    base = record.agent.partition("@")[0]
    if record.seed_version is None:
        doc = default_agent(base)
    else:
        try:
            doc = HarnessStore(root).load(base, str(record.seed_version))
        except (FileNotFoundError, ValueError) as exc:
            raise typer.BadParameter(
                f"cannot reload the recorded seed {base}@v{record.seed_version}: {exc}"
            ) from exc
    if doc.doc_hash != record.seed_doc_hash:
        raise typer.BadParameter(
            f"the recorded seed {base} resolved to doc hash {doc.doc_hash[:12]} but "
            f"this run pinned {record.seed_doc_hash[:12]}; restore the recorded "
            "harness version or start a fresh --run-dir"
        )
    return base, doc


# -- e2b capacity preflight ------------------------------------------------------------------


def _preflight_e2b_capacity(console: Console, *, trial_concurrency: int) -> None:
    """Refuse to start an e2b run that cannot claim the concurrency it asks for.

    E2B caps concurrent sandboxes per account, and a running trial holds
    `E2B_SANDBOXES_PER_TRIAL` of them (harbor's task environment, which lives for its own
    multi-hour timeout; terminus-2 itself runs in this process and needs no sandbox of its
    own). When orphans of an earlier crashed run fill
    the account, every trial fails at sandbox creation with a 429 and the run produces zero
    token spans, which reads exactly like a broken model. So: count what is running, reclaim
    this machine's provable orphans (exact ids whose owning process is gone), and fail with the
    numbers if that is still not enough. The account-wide sweep is never automatic; the message
    names it instead.

    Raises:
        typer.BadParameter: If capacity cannot be measured (missing extra or credential) or
            too few slots are free after reaping the safe class.
    """
    from wmo.optimize.model.rollouts import E2B_SANDBOXES_PER_TRIAL
    from wmo.runtime.harness.e2b_reap import E2B_API_KEY_ENV, is_credential_error

    required = trial_concurrency * E2B_SANDBOXES_PER_TRIAL
    try:
        check_cap_fn = cast(Any, _self.check_capacity)
        check = check_cap_fn(required=required)
    except ImportError as error:
        raise typer.BadParameter(
            f"{error}; the distill config selects harbor.backend = 'e2b'"
        ) from error
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    except Exception as error:  # noqa: BLE001 - a monitoring call must not break a resume
        if is_credential_error(error):
            raise typer.BadParameter(
                f"E2B rejected the sandbox capacity check ({error}); harbor.backend = 'e2b' "
                f"runs every trial in E2B, so set ${E2B_API_KEY_ENV} to an account key (or "
                "switch the distill config to backend = 'local')"
            ) from error
        console.print(
            f"[yellow]warning[/yellow] could not check E2B sandbox capacity "
            f"({type(error).__name__}: {escape(str(error))}); starting anyway"
        )
        return
    if check.reaped:
        console.print(
            f"reaped {check.reaped} orphaned E2B sandbox(es) from dead local runs "
            f"({check.alive_before} -> {check.alive} of {check.cap} in use)"
        )
    if not check.ok:
        raise typer.BadParameter(_capacity_failure_message(check, trial_concurrency))
    console.print(
        f"e2b capacity ok: {check.alive}/{check.cap} sandbox(es) in use, {check.free} free, "
        f"{required} needed ({E2B_SANDBOXES_PER_TRIAL} per trial x "
        f"train.trial_concurrency={trial_concurrency})"
    )


def _capacity_failure_message(check: CapacityCheck, trial_concurrency: int) -> str:
    """The actionable message for a run that cannot get enough sandbox slots."""
    from wmo.optimize.model.rollouts import E2B_SANDBOXES_PER_TRIAL
    from wmo.runtime.harness.e2b_reap import DEFAULT_E2B_SANDBOX_CAP, E2B_SANDBOX_CAP_ENV

    reaped = (
        f" Reaping orphans of dead local runs freed {check.reaped} slot(s) and was not enough."
        if check.reaped
        else " No orphan of a dead local run was left to reclaim."
    )
    affordable = check.free // E2B_SANDBOXES_PER_TRIAL
    lower = f"lower train.trial_concurrency to at most {affordable}, " if affordable >= 1 else ""
    return (
        f"not enough free E2B sandbox slots: {check.alive} of {check.cap} concurrent "
        f"sandboxes are in use, leaving {check.free} free, but this run needs "
        f"{check.required} ({E2B_SANDBOXES_PER_TRIAL} per trial x "
        f"train.trial_concurrency={trial_concurrency}: harbor's task environment)"
        f".{reaped} Either run `wmo e2b reap --stale-minutes 60 --yes` to kill older "
        f"harbor trial sandboxes (account-wide: it can kill another machine's run), {lower}wait "
        f"for the other runs to finish, or raise the account cap (set ${E2B_SANDBOX_CAP_ENV} "
        f"when your cap is not {DEFAULT_E2B_SANDBOX_CAP})"
    )


# -- cost confirmation -----------------------------------------------------------------------


def _print_cost_estimate(console: Console, cfg: DistillConfig, estimate: CostEstimate) -> None:
    """Render the per-meter cost projection; unpriced meters print "unknown"."""
    table = Table(title="Distillation cost estimate")
    table.add_column("Meter", no_wrap=True)
    table.add_column("Tokens", justify="right")
    table.add_column("$/Mtok", justify="right")
    table.add_column("USD", justify="right")
    for line in estimate.lines:
        table.add_row(
            line.meter,
            f"{line.tokens:,}",
            "unknown" if line.price_per_mtok is None else f"{line.price_per_mtok:.3f}",
            "unknown" if line.usd is None else f"{line.usd:.2f}",
        )
    console.print(table)
    cap = (
        f"hard cap budget.max_usd=${cfg.budget.max_usd:.2f}"
        if cfg.budget.max_usd is not None
        else "no budget.max_usd cap"
    )
    warmup = f"{estimate.warmup_episodes} warmup + " if estimate.warmup_episodes > 0 else ""
    console.print(
        f"{estimate.train_episodes} train + {warmup}{estimate.eval_episodes} interim-eval + "
        f"{estimate.baseline_episodes} gate/baseline episode(s); priced total "
        f"${estimate.priced_usd:.2f}; {cap}"
    )


def _confirm_cost(
    console: Console, estimate: CostEstimate, max_usd: float | None, *, yes: bool
) -> None:
    """Confirm the projected spend before anything is run.

    The rule: `--yes` is honored whenever the spend is accountable, meaning
    the estimate is fully priced OR `budget.max_usd` caps the worst case.
    When unpriced meters exist AND `budget.max_usd` is unset, the run's spend
    is unbounded and unaccounted, so interactive confirmation is forced even
    with `--yes`; a non-interactive invocation in that state is rejected with
    instructions (price the meters or set the cap).

    Everything else goes through the shared spend boundary
    (`wmo.cli.consent.require_spend_consent`), so consent is said, never inferred.

    Raises:
        typer.BadParameter: Unbounded spend in a non-interactive session.
        typer.Exit: The user declined (exit code 0), or a non-interactive session was not
            told `--yes` (exit code 2).
    """
    if estimate.unpriced_meters and max_usd is None:
        meters = ", ".join(estimate.unpriced_meters)
        console.print(
            f"[yellow]warning[/yellow] meter(s) {meters} have no \\[pricing] entry and "
            "budget.max_usd is unset: the run's spend is unbounded and unaccounted, "
            "so --yes does not apply here"
        )
        # `can_prompt`, not `console.is_terminal`: the question below reads stdin, so a terminal
        # stdout with a redirected stdin has nobody behind it to answer for unbounded spend.
        if not can_prompt(console):
            raise typer.BadParameter(
                f"cannot start with unbounded spend non-interactively: meter(s) "
                f"{meters} are unpriced and budget.max_usd is unset; add [pricing] "
                "entries for them or set [budget] max_usd in the distill config, "
                "or run it interactively to confirm explicitly"
            )
        try:
            confirmed = Confirm.ask("Proceed with unbounded spend?", default=False)
        except EOFError:
            confirmed = False  # an ended input is not an answer, and never authorizes spend
        if not confirmed:
            raise typer.Exit(0)
        return
    # Consent is said, never inferred: a bounded budget caps the damage but does not grant
    # permission, and this used to start six-figure-token training runs silently.
    cap = f" under the ${max_usd:.2f} budget.max_usd cap" if max_usd is not None else ""
    episodes = (
        estimate.train_episodes
        + estimate.warmup_episodes
        + estimate.eval_episodes
        + estimate.baseline_episodes
    )
    if not require_spend_consent(
        console,
        yes=yes,
        spend=f"~${estimate.priced_usd:.2f} over {episodes} episode(s){cap}",
        command="wmo optimize distill run",
    ):
        raise typer.Exit(0)


# -- completion output -----------------------------------------------------------------------


def _print_result(
    console: Console,
    result: DistillResult,
    store: DistillRunStore,
    *,
    adapters: AdapterStore,
    base_model: str,
) -> None:
    """Print the gate verdict, artifact paths, and the serving handoff snippet."""
    from wmo.optimize.model.store import build_handoff_toml

    gate = result.gate
    color = "green" if gate.accepted else "yellow"
    console.print(f"[{color}]gate[/{color}] {escape(gate.reason)}")
    console.print(
        f"  holdout solve rates: teacher {gate.teacher_solve_rate:.3f}, "
        f"student before {gate.student_before_solve_rate:.3f}, "
        f"after {gate.student_after_solve_rate:.3f}"
    )
    if result.adapter_version is not None:
        console.print(
            f"[green]adapter[/green] [bold]{result.name}[/bold] v{result.adapter_version} "
            f"(champion) -> {adapters.dir_for(result.name) / f'v{result.adapter_version}'}",
            soft_wrap=True,
        )
    else:
        console.print("adapter not promoted; the run dir keeps every artifact for inspection")
    console.print(f"final sampler weights: {result.final_sampler_path}", soft_wrap=True)
    console.print(f"resumable training state: {result.final_state_path}", soft_wrap=True)
    console.print(
        f"spend: ${result.spend.total_usd:.2f} total "
        f"(this session ${result.spend.session_usd:.2f}) -> {result.run_dir}",
        soft_wrap=True,
    )
    try:
        handoff = build_handoff_toml(result.final_sampler_path, base_model=base_model)
    except ValueError as exc:
        console.print(f"[yellow]no handoff snippet[/yellow]: {escape(str(exc))}")
        return
    location = f" (written to {store.handoff_path})" if gate.accepted else ""
    console.print(f"serving handoff{location}:")
    console.print(escape(handoff))


def _maybe_promote(console: Console, result: DistillResult, cfg: DistillConfig, root: str) -> None:
    """Write `[models.agent]` for an accepted adapter, after an explicit confirm.

    The write changes what every subsequent local run and optimization uses as
    the agent model, so it always asks, even under `--yes`; a rejected gate
    skips the write with a warning.
    """
    from wmo.cli.model_roles import load_settings_or_abort
    from wmo.common.config.settings import ModelRole, save_settings, settings_path
    from wmo.optimize.model.store import (
        DEFAULT_TINKER_OPENAI_ENDPOINT,
        STUDENT_CHAT_MAX_TOKENS_FIELD,
    )

    if result.adapter_version is None:
        console.print(
            "[yellow]--promote skipped[/yellow]: the gate rejected this adapter, so "
            "\\[models.agent] was not changed (the handoff snippet above still works "
            "for manual experiments)"
        )
        return
    path = settings_path(root)
    try:
        confirmed = Confirm.ask(
            f"Write models.agent = {result.final_sampler_path} to {path}?",
            default=False,
        )
    except EOFError:
        confirmed = False
    if not confirmed:
        console.print(
            f"skipped writing \\[models.agent]; paste the handoff snippet into {path} when ready"
        )
        return
    settings = load_settings_or_abort(root)
    settings.models.agent = ModelRole(
        provider="openai",
        model=result.final_sampler_path,
        model_type=cfg.student.base_model,
        endpoint=DEFAULT_TINKER_OPENAI_ENDPOINT,
        # A tinker:// path is outside the built-in catalog, so capability resolution would fall
        # back to `max_completion_tokens`, which Tinker's endpoint 400s on. Pin the name it takes.
        chat_max_tokens_field=STUDENT_CHAT_MAX_TOKENS_FIELD,
    )
    save_settings(settings, root)
    console.print(
        f"[green]wrote[/green] \\[models.agent] -> {path} (set WMO_ENDPOINT_API_KEY to "
        "your Tinker API key before running the agent)"
    )


# -- report ------------------------------------------------------------------------------------

# Literal mirrors of `wmo.optimize.model.loop`'s eval keys, kept here so importing this module for
# `--help` does not pull the distill loop's own heavy dependencies.
_REPORT_ROWS: tuple[tuple[str, str], ...] = (
    ("teacher", "baseline-teacher"),
    ("student before", "baseline-student-before"),
    ("student after", "student-after"),
)
"""The three held-out measurements the gate compares, in table order, paired
with the `evals/<key>.json` each one was written to."""


def _load_gate(store: DistillRunStore) -> DistillGateRecord:
    """Read the run's `gate.json`, turning a missing or corrupt file into a usage error."""
    from wmo.optimize.model.gate import DistillGateRecord

    try:
        text = store.gate_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise typer.BadParameter(
            f"no {store.gate_path}: this run has not reached its gate yet (or "
            f"{store.run_dir} is not a distillation run dir). Finish or resume it with "
            "`wmo optimize distill run --run-dir <dir> --resume`"
        ) from exc
    try:
        return DistillGateRecord.model_validate_json(text)
    except ValidationError as exc:
        raise typer.BadParameter(f"cannot load {store.gate_path}: {exc}") from exc


def _load_eval_report(store: DistillRunStore, key: str) -> DistillEvalReport | None:
    """Read one `evals/<key>.json`, or None when the run never wrote it.

    A missing report is normal (an imported baseline is copied in, but an
    aborted run may have none), so the table degrades to the rates gate.json
    already carries rather than failing.
    """
    from wmo.optimize.model.loop import DistillEvalReport

    try:
        text = (store.evals_dir / f"{key}.json").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        return DistillEvalReport.model_validate_json(text)
    except ValidationError as exc:
        raise typer.BadParameter(
            f"cannot load {store.evals_dir / f'{key}.json'}: {exc}; delete the file to "
            "report from gate.json alone"
        ) from exc


def _solve_rate_table(store: DistillRunStore, gate: DistillGateRecord) -> Table:
    """The teacher / student-before / student-after held-out comparison."""
    rates = (
        gate.teacher_solve_rate,
        gate.student_before_solve_rate,
        gate.student_after_solve_rate,
    )
    table = Table(title=f"Held-out solve rates ({store.run_dir})")
    table.add_column("Measurement", no_wrap=True)
    # Fold rather than ellipsize. A tinker:// sampler path is wider than the ~22 columns
    # this cell gets at an 80-column terminal, and rich's default would truncate it to
    # `tinker://weights/pi...`, silently dropping the identity of the artifact. Folding
    # keeps every character; the copyable form is printed on its own line below.
    table.add_column("Model", overflow="fold")
    table.add_column("Solve", justify="right")
    table.add_column("Graded", justify="right")
    table.add_column("Executed", justify="right")
    table.add_column("Scaffold", justify="right")
    for (label, key), rate in zip(_REPORT_ROWS, rates, strict=True):
        eval_report = _load_eval_report(store, key)
        if eval_report is None:
            table.add_row(label, "unknown", f"{rate:.3f}", "-", "-", "-")
            continue
        graded = (
            f"{eval_report.graded_solve_rate:.3f}" if eval_report.graded_trials else "unmeasured"
        )
        table.add_row(
            label,
            eval_report.provider_model,
            f"{rate:.3f}",
            graded,
            f"{eval_report.executed_trials}/{eval_report.trials}",
            f"{eval_report.scaffold_loss_rate:.0%}",
        )
    return table


def _print_trained_artifact(console: Console, store: DistillRunStore) -> None:
    """Print the sampler path the student-after numbers came from, on its own line.

    The table names it too, but a `tinker://` path is wider than the cell it gets at an
    80-column terminal, so there it folds across lines. This line is the copyable one: it
    is what you paste into a pool entry or a follow-on run's `init_from_state`.

    Args:
        console: Where to print.
        store: The run store to read the student-after eval report from.
    """
    after = _load_eval_report(store, "student-after")
    if after is None or not after.provider_model:
        return
    console.print(f"trained artifact: {escape(after.provider_model)}")


def _print_paired_delta(console: Console, store: DistillRunStore, gate: DistillGateRecord) -> None:
    """Print what training moved, on the same holdout split the gate read."""
    binary = gate.student_after_solve_rate - gate.student_before_solve_rate
    console.print(f"paired delta (after - before): {binary:+.3f} solve rate")
    before = _load_eval_report(store, "baseline-student-before")
    after = _load_eval_report(store, "student-after")
    if before is not None and after is not None and before.graded_trials and after.graded_trials:
        graded = after.graded_solve_rate - before.graded_solve_rate
        console.print(f"  graded (same trials at test resolution): {graded:+.3f}")
    fraction = (
        gate.student_after_solve_rate / gate.teacher_solve_rate
        if gate.teacher_solve_rate > 0
        else None
    )
    reached = "unmeasurable (teacher solved nothing)" if fraction is None else f"{fraction:.3f}"
    verdict = "passed" if gate.accepted else "FAILED"
    console.print(
        f"  after / teacher: {reached} against gate minimum "
        f"{gate.min_teacher_fraction:.2f}; gate {verdict}"
    )


def _read_metrics(console: Console, store: DistillRunStore) -> list[JsonObject]:
    """Every metrics row, surviving the half-written last line an aborted run leaves.

    `report` advertises itself as safe on a live or aborted run dir, so a torn final row (all a
    run killed mid-append can leave behind) must not end the command: drop it, say so, and
    report what is complete. Every other shape of damage -- a broken row above the last one, or
    a last line that parses into something other than a JSON object, which no truncated append
    can produce -- means the file lost or gained content, so it stays an error; it is just a
    usage error now, the way `_load_gate` and `_load_eval_report` already treat the same class
    of damage, rather than a traceback.

    Args:
        console: Where to print the note about a dropped final line.
        store: The run store to read `metrics.jsonl` from.

    Returns:
        Every complete row, in append order.

    Raises:
        typer.BadParameter: If the damage is anything but a half-written last line.
    """
    try:
        return store.read_metrics()
    except ValueError:
        pass  # the tolerant read below decides whether the damage is only the torn tail
    try:
        rows = store.read_metrics(tolerate_partial_tail=True)
    except ValueError as fatal:
        raise typer.BadParameter(str(fatal)) from fatal
    console.print(
        f"[yellow]note[/yellow] ignoring a half-written last line in "
        f"{escape(str(store.metrics_path))} (a run killed mid-append leaves one); "
        f"reporting the {len(rows)} complete row(s)"
    )
    return rows


def _print_training_summary(console: Console, store: DistillRunStore) -> None:
    """Print the last training row's health metrics, or say the run trained nothing.

    Turns per episode is deliberately absent: nothing in the run dir records
    it. `mean_generation_tokens` is the per-episode series the loop does
    measure (sampled tokens, pooled over the batch's span-bearing episodes).
    """
    rows = [row for row in _read_metrics(console, store) if row.get("phase") is None]
    if not rows:
        console.print("no training step recorded in metrics.jsonl")
        return
    last = rows[-1]
    step = _row_int(last, "step")
    parts = [f"{len(rows)} training step(s) recorded"]
    for label, key, spec in (
        ("reverse KL/token", "reverse_kl_per_token", ".4f"),
        ("entropy ratio", "entropy_ratio", ".2f"),
        ("tokens/episode", "mean_generation_tokens", ".0f"),
        ("tokens/episode ratio", "generation_tokens_ratio", ".2f"),
    ):
        value = _row_float(last, key)
        if value is not None:
            parts.append(f"{label} {value:{spec}}")
    spent = _row_float(last, "cumulative_usd")
    if spent is not None:
        parts.append(f"${spent:.2f} spent")
    head = "training" if step is None else f"training (last row step {step})"
    console.print(f"{head}: {', '.join(parts)}")


def _row_float(row: JsonObject, key: str) -> float | None:
    """One metrics-row number, or None when absent or not numeric."""
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _row_int(row: JsonObject, key: str) -> int | None:
    """One metrics-row integer, or None when absent or not an integer."""
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def __getattr__(name: str) -> object:
    """Lazy module attribute resolution for deferred CLI imports."""
    if name == "run_distillation":
        from wmo.optimize.model.loop import run_distillation

        return run_distillation
    if name == "check_capacity":
        from wmo.runtime.harness.e2b_reap import check_capacity

        return check_capacity
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


