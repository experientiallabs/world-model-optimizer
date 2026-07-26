"""`wmh optimize harness <agent> harbor --mode distill`: the CLI face of on-policy distillation.

Kept out of `harness_app.py` the way eval's closed-loop half lives in
`eval_closed_loop.py`: `optimize()` validates the flag surface and routes here
early when distill mode is selected. This module owns the distill run's CLI
lifecycle: load and pin the run inputs (config, task splits, seed harness),
project the run cost into a confirmation table, drive `run_distillation` with
progress rendering, and print the gate verdict plus the serving handoff. The
optional `--promote` step writes `[models.agent]` through the settings save
path after an explicit confirmation.

Run-dir pinning mirrors `run-config.json` in the harbor search flow: a fresh
run records its CLI-level inputs in `distill-run.json` (task splits, backend,
the exact seed version and doc hash), and a resume reuses that record instead
of live flags, rejecting explicit flags that conflict with it. The distill
config itself is snapshotted by the run store as `config.toml`, which is what
a bare `--resume` (no `--distill-config`) loads.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import typer
from pydantic import BaseModel, ConfigDict, ValidationError
from rich.console import Console
from rich.markup import escape
from rich.prompt import Confirm
from rich.table import Table

from wmh.agents.default import default_agent
from wmh.config.settings import ModelRole, load_settings, save_settings, settings_path
from wmh.config.store import validate_name
from wmh.distill.config import DistillConfig, load_distill_config
from wmh.distill.cost import CostEstimate, estimate_run_cost
from wmh.distill.loop import (
    DistillBudgetError,
    DistillProgress,
    DistillResult,
    run_distillation,
)
from wmh.distill.rollouts import E2B_SANDBOXES_PER_TRIAL
from wmh.distill.store import (
    DEFAULT_TINKER_OPENAI_ENDPOINT,
    AdapterStore,
    DistillRunStore,
    build_handoff_toml,
)
from wmh.harness.doc import HarnessDoc
from wmh.harness.e2b_reap import (
    DEFAULT_E2B_SANDBOX_CAP,
    E2B_API_KEY_ENV,
    E2B_SANDBOX_CAP_ENV,
    CapacityCheck,
    check_capacity,
    is_credential_error,
)
from wmh.harness.population import write_json_atomic
from wmh.harness.store import HarnessStore

DISTILL_RUN_RECORD = "distill-run.json"
"""The CLI-level pin file inside the run dir (see `DistillCliRunRecord`)."""

_PI_NODE_RUNTIME = "pi-node"


class DistillCliRunRecord(BaseModel):
    """The CLI inputs pinned into `distill-run.json` when a distill run starts.

    A resume command carries only `--run-dir` (that is what a budget abort
    prints), so everything else the CLI resolved at start is recorded here and
    reloaded on resume; explicit flags that conflict with the record are
    rejected instead of silently changing what is being trained or gated.
    """

    model_config = ConfigDict(frozen=True)

    agent: str
    """The AGENT argument exactly as given (may carry an @ref)."""

    backend: Literal["local", "e2b"]
    seed_version: int | None
    """The stored seed version; None means the built-in default agent."""

    seed_doc_hash: str
    """The resolved seed document's hash; a resume must re-resolve to it."""

    train_task_ids: tuple[str, ...]
    holdout_task_ids: tuple[str, ...]


def run_distill(
    console: Console,
    *,
    agent_name: str | None,
    distill_config_path: str | None,
    task_ids_path: str | None,
    holdout_task_ids_path: str | None,
    run_dir: str | None,
    backend: str | None,
    resume: bool,
    yes: bool,
    promote: bool,
    root: str,
) -> None:
    """Run (or resume) one on-policy distillation from the CLI.

    Args:
        console: The CLI's rich console (product output goes through it).
        agent_name: The AGENT argument; the literal 'pi' is the built-in
            default agent, 'name@ref' seeds from the harness store.
        distill_config_path: The per-run distill TOML; required to start a
            fresh run. On resume None loads the run dir's config.toml
            snapshot, and an explicit path wins over it (the documented
            budget-abort recovery is editing budget.max_usd and resuming).
        task_ids_path: JSON array of train task ids; required to start.
        holdout_task_ids_path: JSON array of holdout task ids; required to
            start. Baselines and the gate are measured here.
        run_dir: The run's durable state directory; always required.
        backend: An explicit `--backend` override for the config's
            harbor.backend, or None when the flag was not given.
        resume: Continue the run recorded in `run_dir`.
        yes: Skip the cost confirmation (see `_confirm_cost` for the one
            case where confirmation is forced anyway).
        promote: After an accepted gate, offer to write `[models.agent]`
            pointing at the distilled adapter (explicit confirmation).
        root: The project dir (harness store, adapter store, settings).

    Raises:
        typer.BadParameter: On any invalid or conflicting input; the message
            names the flag and what to do.
        typer.Exit: When the user declines a confirmation (code 0) or the
            run fails/aborts (code 1).
    """
    # Deferred import: harness_app routes to this module at module scope, so
    # importing its helpers back at module scope would be a circular import.
    from wmh.cli.harness_app import DEFAULT_SEED_AGENT, _load_harbor_task_ids

    if agent_name is None:
        raise typer.BadParameter(
            "provide the agent NAME whose harness the pi trials run (the literal "
            f"{DEFAULT_SEED_AGENT!r} is the built-in default agent): "
            "`wmh optimize harness pi harbor --mode distill --distill-config run.toml "
            "--task-ids train.json --holdout-task-ids holdout.json --run-dir <dir>`"
        )
    backend_override: Literal["local", "e2b"] | None
    if backend is None:
        backend_override = None
    elif backend == "e2b":
        backend_override = "e2b"
    elif backend == "local":
        backend_override = "local"
    else:
        raise typer.BadParameter(f"unknown --backend {backend!r}; choose local or e2b")
    if run_dir is None:
        raise typer.BadParameter(
            "--run-dir is required for --mode distill: it holds all durable run "
            "state (config snapshot, metrics, checkpoints, rollout artifacts)"
        )

    run_path = Path(run_dir)
    record_path = run_path / DISTILL_RUN_RECORD
    store = DistillRunStore(run_path)
    seed_version: int | None
    if resume:
        record = _load_record(record_path)
        _reject_resume_conflicts(
            record,
            agent_name=agent_name,
            backend=backend_override,
            task_ids_path=task_ids_path,
            holdout_task_ids_path=holdout_task_ids_path,
            load_task_ids=_load_harbor_task_ids,
        )
        train_ids = record.train_task_ids
        holdout_ids = record.holdout_task_ids
        cfg = _load_config(
            Path(distill_config_path) if distill_config_path is not None else store.config_path
        )
        base, seed_doc = _pinned_seed_doc(root, record, DEFAULT_SEED_AGENT)
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
                ("--distill-config", distill_config_path),
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
        assert distill_config_path is not None  # narrowed by the missing check
        assert task_ids_path is not None and holdout_task_ids_path is not None
        cfg = _load_config(Path(distill_config_path))
        train_ids = _load_harbor_task_ids(Path(task_ids_path))
        holdout_ids = _load_harbor_task_ids(Path(holdout_task_ids_path))
        base, seed_doc, seed_version = _resolve_seed_doc(root, agent_name, DEFAULT_SEED_AGENT)
        effective_backend = backend_override if backend_override is not None else cfg.harbor.backend

    overlap = sorted(set(train_ids) & set(holdout_ids))
    if overlap:
        raise typer.BadParameter(
            f"task id(s) {', '.join(overlap)} appear in BOTH --task-ids and "
            "--holdout-task-ids; the gate is only meaningful on tasks the student "
            "never trained on, so make the splits disjoint"
        )
    if effective_backend != cfg.harbor.backend:
        cfg = cfg.model_copy(
            update={"harbor": cfg.harbor.model_copy(update={"backend": effective_backend})}
        )
    runtime_kind = seed_doc.runtime_kind()
    if runtime_kind != _PI_NODE_RUNTIME:
        raise typer.BadParameter(
            f"distillation rollouts drive the pi agent through harbor trials, but "
            f"harness {agent_name!r} has runtime kind {runtime_kind!r}; seed from a "
            f"pi-node harness (the built-in {DEFAULT_SEED_AGENT!r} agent, or a "
            "version optimized from it)"
        )
    template_path = Path(cfg.harbor.job_template)
    if not template_path.is_file():
        raise typer.BadParameter(
            f"harbor.job_template {template_path} does not exist; point the distill "
            "config's [harbor] job_template at the harbor JobConfig YAML/JSON the "
            "rollouts should run"
        )
    if effective_backend == "e2b":
        _preflight_e2b_capacity(console, trial_concurrency=cfg.train.trial_concurrency)

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
            agent=agent_name,
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
        result = run_distillation(
            base,
            cfg,
            seed_doc,
            list(train_ids),
            list(holdout_ids),
            run_path,
            resume=resume,
            on_progress=_on_progress,
            adapter_store=AdapterStore(root),
            # Resume commands must print the agent string as typed (it may carry
            # an @ref that `base` strips), or the printed command would trip the
            # CLI's resume conflict check.
            cli_agent=agent_name,
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


def _load_config(path: Path) -> DistillConfig:
    """Load the distill TOML, turning load failures into usage errors."""
    try:
        return load_distill_config(path)
    except FileNotFoundError as exc:
        raise typer.BadParameter(
            f"{exc} (a fresh run needs --distill-config; a resume reads the run "
            "dir's config.toml snapshot)"
        ) from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


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
    agent_name: str,
    backend: str | None,
    task_ids_path: str | None,
    holdout_task_ids_path: str | None,
    load_task_ids: Callable[[Path], tuple[str, ...]],
) -> None:
    """Reject explicit flags that conflict with the recorded run inputs."""
    conflicts: list[str] = []
    if agent_name != record.agent:
        conflicts.append(f"AGENT {agent_name!r} != recorded {record.agent!r}")
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


def _resolve_seed_doc(
    root: str, agent_ref: str, default_seed_name: str
) -> tuple[str, HarnessDoc, int | None]:
    """Resolve the AGENT positional to the harness the pi trials run.

    Mirrors the harbor search's seed protocol: the bare default-agent literal
    is ALWAYS the built-in agent; 'name@ref' loads a stored version. Returns
    the base name, the document, and the resolved store version (None for the
    built-in seed) so a resume can pin exactly what the run started from.
    """
    base, _, ref = agent_ref.partition("@")
    try:
        validate_name(base)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if base == default_seed_name and not ref:
        return base, default_agent(base), None
    try:
        doc = HarnessStore(root).load(base, ref or None)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(
            f"{exc}; the built-in default agent is the literal {default_seed_name!r}"
        ) from exc
    return base, doc, doc.version


def _pinned_seed_doc(
    root: str, record: DistillCliRunRecord, default_seed_name: str
) -> tuple[str, HarnessDoc]:
    """Re-resolve the recorded seed for a resume, never a live movable ref.

    The record pins the exact version and doc hash, so champion movement (or
    any store edit) between sessions cannot silently change which harness the
    remaining trials run.
    """
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
    required = trial_concurrency * E2B_SANDBOXES_PER_TRIAL
    try:
        check = check_capacity(required=required)
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
        f".{reaped} Either run `wmh e2b reap --stale-minutes 60 --yes` to kill older "
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
    offpolicy = (
        f"{estimate.offpolicy_episodes} off-policy + " if estimate.offpolicy_episodes > 0 else ""
    )
    console.print(
        f"{estimate.train_episodes} train + {offpolicy}{warmup}"
        f"{estimate.eval_episodes} interim-eval + "
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

    Raises:
        typer.BadParameter: Unbounded spend in a non-interactive session.
        typer.Exit: The user declined (exit code 0).
    """
    if estimate.unpriced_meters and max_usd is None:
        meters = ", ".join(estimate.unpriced_meters)
        console.print(
            f"[yellow]warning[/yellow] meter(s) {meters} have no \\[pricing] entry and "
            "budget.max_usd is unset: the run's spend is unbounded and unaccounted, "
            "so --yes does not apply here"
        )
        if not console.is_terminal:
            raise typer.BadParameter(
                f"cannot start with unbounded spend non-interactively: meter(s) "
                f"{meters} are unpriced and budget.max_usd is unset; add [pricing] "
                "entries for them or set [budget] max_usd in the distill config, "
                "or run at a TTY to confirm explicitly"
            )
        if not Confirm.ask("Proceed with unbounded spend?", default=False):
            raise typer.Exit(0)
        return
    if console.is_terminal and not yes and not Confirm.ask("Proceed?", default=True):
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
    try:
        settings = load_settings(root)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    settings.models.agent = ModelRole(
        provider="openai",
        model=result.final_sampler_path,
        model_type=cfg.student.base_model,
        endpoint=DEFAULT_TINKER_OPENAI_ENDPOINT,
    )
    save_settings(settings, root)
    console.print(
        f"[green]wrote[/green] \\[models.agent] -> {path} (set WMH_ENDPOINT_API_KEY to "
        "your Tinker API key before running the agent)"
    )
