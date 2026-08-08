"""`wmo runs`: see and steer the runs this machine (and its siblings) are feeding.

The panel's contents, in a terminal. A long grid or a staged optimization reports itself while it
works (`wmo.optimize.telemetry.hooks`), and these commands read that back:

    wmo runs list                       what is running, how far in, what it has spent
    wmo runs show jt/grid-c2/identity   stages, per-candidate cells, pending commands
    wmo runs tail jt/grid-c2/identity   the live event log, resumable
    wmo runs stop jt/grid-c2/identity   ask the process feeding it to stop cleanly

`backfill` is the other direction: a run that finished (or died) before it was ever reported is
replayed from its own artifacts, so the panel gains history it never watched happen.

Every read is scoped to the organization the saved credential belongs to, and `--json` is there
because an agent reading this output is as likely as a person.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape
from rich.table import Table

import wmo.cli.runs_app as _self
from wmo.common.config import ARTIFACT_DIR
from wmo.runtime.platform.credentials import credentials_path
from wmo.runtime.runs.schema import (
    LEDGER_LINE,
    LOG_LINE,
    RunEvent,
    RunEventType,
    RunKind,
    RunStatus,
    grid_arm_external_id,
    grid_relpath,
    is_terminal_status,
    pipeline_external_id,
)

if TYPE_CHECKING:
    from wmo.common.core.types import JsonObject
    from wmo.runtime.platform.client import PlatformError
    from wmo.runtime.runs.reader import CellStats, EventRow, RunDetail, RunsReader, RunSummary
    from wmo.runtime.runs.schema import RunEvent

log = logging.getLogger(__name__)

_console = Console()

runs_app = typer.Typer(
    help="See and steer optimization runs: list, show, tail, stop, retry, backfill.",
    no_args_is_help=True,
)

COHORT_FILE = "cohort.json"
"""What makes a directory a grid: the cohort every cell in it was bought under."""

POLL_INTERVAL_S = 2.0
"""Pause between polls when following one event type. Matches the server's safe-frontier lag, so a
poll lands where the next matching event is about to be servable rather than spinning."""

LEDGER_FILE = "ledger.jsonl"
MANIFEST_RELPATH = Path("optimize") / "optimize-run.json"

_NOT_CONNECTED = (
    "not connected to a platform, so there is nothing to read; run `wmo login` "
    "(or set WMO_PLATFORM_TOKEN and WMO_PLATFORM_ORG)"
)

_STATUS_STYLE = {
    RunStatus.RUNNING: "cyan",
    RunStatus.COMPLETED: "green",
    RunStatus.FAILED: "red",
    RunStatus.STOPPED: "yellow",
}

# Listed from the enums rather than retyped, so the help can never drift from what the platform
# actually accepts.
_STATUSES = ", ".join(status.value for status in RunStatus)
_KINDS = ", ".join(kind.value for kind in RunKind)


_ORG = typer.Option(
    "--org",
    help="Organization id or slug to read (default: the login's, or $WMO_PLATFORM_ORG).",
)


def _reader(org: str | None = None) -> RunsReader:
    """The org-scoped reader, or a clean usage error naming the fix."""
    from wmo.runtime.platform.client import PlatformError

    try:
        reader_cls = cast(Any, _self.RunsReader)
        reader = reader_cls.open(org=org)
    except PlatformError as error:
        raise _failed("Could not resolve the organization", error) from error
    if reader is None:
        raise typer.BadParameter(f"{_NOT_CONNECTED} (credential file: {credentials_path()})")
    return reader


def _status(value: str) -> str:
    return f"[{_STATUS_STYLE.get(value, 'white')}]{value}[/]"


def _progress(summary: RunSummary) -> str:
    """A run's progress as one cell of the table: done/total, and how many scored."""
    progress = summary.progress
    done = progress.get("done")
    total = progress.get("total")
    scored = progress.get("scored")
    stage = progress.get("stage")
    if done is None and stage is None:
        return "-"
    shown = f"{done}/{total}" if total else f"{done}" if done is not None else "-"
    if scored is not None:
        shown = f"{shown} ({scored} scored)"
    if stage is not None:
        shown = f"{shown} @ {stage}" if done is not None else str(stage)
    return shown


@runs_app.command("list")
def list_runs(
    status: Annotated[
        str | None,
        typer.Option("--status", help=f"Only one state: {_STATUSES}."),
    ] = None,
    kind: Annotated[
        str | None,
        typer.Option("--kind", help=f"Only one kind: {_KINDS}."),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="How many runs to show.")] = 20,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
    org: Annotated[str | None, _ORG] = None,
) -> None:
    """List this organization's runs, newest first."""
    from wmo.runtime.platform.client import PlatformError

    with _reader(org) as reader:
        try:
            page = reader.list_runs(status=status, kind=kind, limit=limit)
        except PlatformError as error:
            raise _failed("Could not list runs", error) from error
        except ValidationError as error:
            raise _shape_error("the run list", error) from error
    if as_json:
        _console.print_json(
            json.dumps({"runs": [run.model_dump(mode="json") for run in page.runs]})
        )
        return
    if not page.runs:
        _console.print("No runs reported yet.")
        return
    table = Table(show_header=True, header_style="bold")
    for column in ("run", "kind", "status", "progress", "spend", "started"):
        table.add_column(column)
    for run in page.runs:
        table.add_row(
            escape(run.external_id),
            escape(run.kind),
            _status(run.status),
            _progress(run),
            f"${run.spend_usd:,.2f}",
            (run.started_at or "")[:19],
        )
    _console.print(table)
    if page.next_cursor is not None:
        _console.print(f"[dim]more runs available; showing the newest {len(page.runs)}[/dim]")


@runs_app.command("show")
def show_run(
    external_id: Annotated[str, typer.Argument(help="The run's name, e.g. jt/grid-c2/identity.")],
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
    org: Annotated[str | None, _ORG] = None,
) -> None:
    """Show one run: where it is, what it spent, its stages and its cells."""
    from wmo.runtime.platform.client import PlatformError

    with _reader(org) as reader:
        try:
            detail = reader.get_run(external_id)
        except PlatformError as error:
            raise _failed(f"Could not read run {external_id!r}", error) from error
        except ValidationError as error:
            raise _shape_error(f"run {external_id!r}", error) from error
    if as_json:
        _console.print_json(json.dumps(detail.model_dump(mode="json")))
        return
    _print_detail(detail)


def _print_detail(detail: RunDetail) -> None:
    """Render a run the way the panel's detail page reads, top down."""
    run = detail.run
    _console.print(f"[bold]{escape(run.external_id)}[/bold]  {_status(run.status)}  ({run.kind})")
    for label, value in (
        ("benchmark", run.benchmark),
        ("arm", run.arm),
        ("started", run.started_at),
        ("last heartbeat", run.heartbeat_at),
        ("finished", run.finished_at),
    ):
        if value:
            _console.print(f"  {label}: {value}")
    _console.print(f"  progress: {_progress(run)}")
    # candidate + world model, never plus compressor: the compressor's inference is already folded
    # into the candidate side, so adding it would bill it twice.
    _console.print(
        f"  spend: ${run.spend_usd:,.4f} "
        f"(candidate ${run.candidate_usd or 0.0:,.4f}, world model ${run.wm_usd or 0.0:,.4f}, "
        f"of which compressor ${run.compressor_usd or 0.0:,.4f})"
    )
    _console.print(f"  events: {detail.event_count}")
    if run.error:
        _console.print(f"  [red]error:[/red] {escape(run.error)}")

    if detail.stages:
        table = Table(show_header=True, header_style="bold", title="stages", title_justify="left")
        for column in ("stage", "status", "spend", "completed"):
            table.add_column(column)
        for stage in detail.stages:
            spend = (stage.candidate_usd or 0.0) + (stage.wm_usd or 0.0)
            table.add_row(
                escape(stage.stage),
                _status(stage.status),
                f"${spend:,.4f}",
                (stage.completed_at or "")[:19],
            )
        _console.print(table)

    if detail.cell_stats:
        _console.print(_cells_table(detail.cell_stats))

    if detail.pending_control:
        _console.print("[bold]pending commands[/bold]")
        for control in detail.pending_control:
            note = f" - {escape(control.note)}" if control.note else ""
            _console.print(f"  {escape(control.command)} ({escape(control.status)}){note}")


def _cells_table(stats: tuple[CellStats, ...]) -> Table:
    """Per-candidate cells: how many ran, how many scored, how many failed, mean reward, cost.

    `unpriced` sits beside the cost on purpose, and is never folded into it: a total summed over a
    partially priced matrix under-reports by however many cells had no verified price, and a cost
    figure that hides how complete it is is the one number a reader will quote without checking.
    """
    table = Table(show_header=True, header_style="bold", title="cells", title_justify="left")
    for column in ("candidate", "cells", "scored", "errors", "mean reward", "cost", "unpriced"):
        table.add_column(column)
    for row in stats:
        table.add_row(
            escape(row.model),
            str(row.cell_count),
            str(row.scored_count),
            str(row.error_count) if row.error_count else "-",
            f"{row.reward_mean:.3f}" if row.reward_mean is not None else "-",
            f"${row.cost_usd_total:,.4f}" if row.cost_usd_total is not None else "-",
            f"[yellow]{row.unpriced_count}[/yellow]" if row.unpriced_count else "-",
        )
    return table


@runs_app.command("tail")
def tail_run(
    external_id: Annotated[str, typer.Argument(help="The run's name.")],
    from_pos: Annotated[
        int,
        typer.Option(
            "--from-pos",
            help="Resume the log after this stream position (0 opens at the end of the log).",
        ),
    ] = 0,
    event_type: Annotated[
        str | None, typer.Option("--type", help=f"Only one event type, e.g. {LEDGER_LINE}.")
    ] = None,
    backlog: Annotated[
        int, typer.Option("--backlog", help="Events of history to print before following.")
    ] = 20,
) -> None:
    """Follow a run's event log until it finishes, printing one line per event.

    A live run's log deliberately trails real time by about two seconds (the server serves only
    positions nothing can arrive behind), and a finished run's tail ends on its own rather than
    hanging. Ctrl-C stops following and changes nothing about the run.
    """
    from wmo.runtime.platform.client import PlatformError

    with _reader() as reader:
        try:
            cursor = _print_backlog(
                reader, external_id, from_pos=from_pos, backlog=backlog, event_type=event_type
            )
            if event_type is None:
                _follow(reader, external_id, cursor)
            else:
                _poll(reader, external_id, cursor, event_type)
        except PlatformError as error:
            raise _failed(f"Could not tail run {external_id!r}", error) from error
        except KeyboardInterrupt:
            _console.print("\n[dim]stopped following; the run is untouched[/dim]")


def _print_backlog(
    reader: RunsReader,
    external_id: str,
    *,
    from_pos: int,
    backlog: int,
    event_type: str | None = None,
) -> int:
    """Print the history a viewer wants before following, and return the cursor to resume at.

    Carries `--type` too: a backlog of every event type followed by a single-type tail would look
    like the filter had been ignored and then applied halfway down the screen.
    """
    if backlog <= 0:
        # Explicitly none. Falling through would open at the END of the log with `tail=True`, which
        # is the opposite of what asking for no history means.
        return from_pos
    page = reader.list_events(
        external_id,
        after_pos=from_pos,
        limit=backlog,
        tail=from_pos == 0,
        event_type=event_type,
    )
    for event in page.events:
        _console.print(_event_line(event))
    return page.last_pos


def _follow(reader: RunsReader, external_id: str, cursor: int) -> None:
    """Stream events as they arrive, over the resumable SSE tail."""
    for event in reader.tail(external_id, after_pos=cursor):
        _console.print(_event_line(event))


def _poll(reader: RunsReader, external_id: str, cursor: int, event_type: str) -> None:
    """Follow one event type by paging, since the stream carries every type.

    Filtering client-side over the stream would be the obvious alternative and is worse: the
    server can filter in the database, so a `--type ledger.line` tail reads the handful of events
    it wants instead of draining a whole grid's cell batches to discard them.
    """
    while True:
        page = reader.list_events(external_id, after_pos=cursor, event_type=event_type)
        for event in page.events:
            _console.print(_event_line(event))
        cursor = page.last_pos
        # The shared vocabulary, not `!= running`: a future status meaning still-writing (paused,
        # resuming) would make the inversion end a LIVE tail early and silently drop the rest of
        # the run. `is_terminal_status` errs the other way (see its docstring).
        if is_terminal_status(reader.get_run(external_id).run.status):
            return
        _sleep_between_polls()


def _sleep_between_polls() -> None:
    """Wait out one safe-frontier lag before asking again."""
    time.sleep(POLL_INTERVAL_S)


def _event_line(event: EventRow) -> str:
    """One event as one readable line: position, clock, type, and its own summary."""
    return (
        f"[dim]{event.pos:>6}[/dim] {event.ts[:19]} [bold]{event.type}[/bold] "
        f"{escape(_summarize(event))}"
    )


def _summarize(event: EventRow) -> str:
    """The part of a payload worth a terminal line, per event type."""
    payload = event.payload
    if event.type == RunEventType.CELL_BATCH:
        cells = payload.get("cells")
        count = len(cells) if isinstance(cells, list) else 0
        return f"{count} cell(s)"
    if event.type == RunEventType.HEARTBEAT:
        progress = payload.get("progress")
        spend = payload.get("spend")
        shown = []
        if isinstance(progress, dict):
            shown.append(f"done={progress.get('done')} scored={progress.get('scored')}")
        if isinstance(spend, dict):
            candidate = spend.get("candidate_usd") or 0.0
            wm = spend.get("wm_usd") or 0.0
            if isinstance(candidate, int | float) and isinstance(wm, int | float):
                shown.append(f"spend=${candidate + wm:,.2f}")
        return " ".join(shown)
    if event.type == LEDGER_LINE:
        note = str(payload.get("note") or "")
        return (
            f"{payload.get('event')} chunk={payload.get('chunk')} cells={payload.get('cells')} "
            f"scored={payload.get('scored')}{f' {note}' if note else ''}"
        )
    if event.type in (RunEventType.RUN_STATUS, RunEventType.STAGE_UPSERT, RunEventType.RUN_META):
        keys = ("status", "stage", "kind", "error", "reason")
        return " ".join(f"{key}={payload[key]}" for key in keys if payload.get(key) is not None)
    if event.type == LOG_LINE:
        return f"[{payload.get('level')}] {payload.get('line')}"
    return ""


@runs_app.command("stop")
def stop_run(
    external_id: Annotated[str, typer.Argument(help="The run's name.")],
) -> None:
    """Ask the process feeding a run to stop cleanly at its next safe boundary.

    Delivery is pull-based: the command waits until that process next reports in, and a grid stops
    between chunks so nothing half-measured is thrown away. Nothing here kills a process, and the
    run's status does not change until the runner itself says it stopped.
    """
    _command(external_id, "stop", None, "stop")


@runs_app.command("retry")
def retry_run(
    external_id: Annotated[str, typer.Argument(help="The run's name.")],
    chunk: Annotated[
        int | None, typer.Option("--chunk", help="Only re-score this chunk's unscored cells.")
    ] = None,
) -> None:
    """Ask a run to re-measure the cells it left unscored.

    A runner that owns its own retry policy answers this with a reasoned refusal rather than
    obeying it (the tau grid runner already retries every transient failure exactly once, and
    records the spent retries so a restart cannot buy a second). Read `wmo runs show` after
    issuing it: a rejected command carries the note explaining what to do instead.
    """
    args: JsonObject | None = None if chunk is None else {"chunk": chunk}
    _command(external_id, "retry_unscored", args, "retry")


def _command(external_id: str, command: str, args: JsonObject | None, label: str) -> None:
    """Queue one control command and report what the platform recorded."""
    from wmo.runtime.platform.client import PlatformError

    with _reader() as reader:
        try:
            control = reader.request_control(external_id, command, args)
        except PlatformError as error:
            raise _failed(f"Could not {label} run {external_id!r}", error) from error
    _console.print(
        f"queued [bold]{escape(control.command)}[/bold] for {escape(external_id)} "
        f"(id {control.id}, {control.status}). It takes effect when the process feeding the run "
        "next reports in; "
        "`wmo runs show` lists it until then."
    )


@runs_app.command("backfill")
def backfill(
    path: Annotated[
        Path,
        typer.Argument(
            help="A grid directory (holding cohort.json), a world-model directory, or an "
            "optimize-run.json manifest.",
        ),
    ],
    arm: Annotated[
        str | None, typer.Option("--arm", help="One arm of a grid directory (default: every arm).")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Write the events as JSONL instead of pushing them.")
    ] = False,
    out: Annotated[
        Path | None, typer.Option("--out", help="Where --dry-run writes (default: stdout).")
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Push even when the run already has events. Only for a run whose earlier events "
            "were lost; a run that reported itself live is already complete.",
        ),
    ] = False,
    name: Annotated[
        str | None,
        typer.Option(
            "--name",
            help="Name the run instead of deriving it from the path. For artifacts that have been "
            "MOVED (copied out of .wmo, restored from a bundle, staged on another machine), where "
            "the path no longer says which run they belong to. For a grid directory this is the "
            "grid prefix and the arm still appends (--name jt/grid-c2 gives jt/grid-c2/<arm>); for "
            "a manifest it is the whole run name.",
        ),
    ] = None,
) -> None:
    """Replay a finished or interrupted run from its artifacts, so the panel has its history.

    Nothing is inferred that the artifacts do not say. Every timestamp is the one on disk (a ledger
    line's clock, a cohort's creation, a merge's `merged_at`, a stage's `completed_at`), so a
    backfill run reads as what happened when it happened rather than when it was replayed, and
    re-running it produces byte-identical events. That is what makes a repeat free: the platform
    keys events on the emitter's seq and discards ones it already holds.

    The run's name comes from where the artifacts live (a grid's path under `.wmo`, a manifest's own
    `world_model`), so replaying artifacts in place needs nothing else. `--name` is for artifacts
    that have MOVED, where the path would name a run nobody recognizes:

        wmo runs backfill .wmo/jt/grid-c2 --arm identity
        wmo runs backfill .wmo/models/tau-bench --dry-run --out events.jsonl
        wmo runs backfill /restored/artifacts --name jt/grid-c2   # moved out of .wmo
    """
    plans = _plan_backfill(path, arm, name)
    if dry_run:
        _write_jsonl(plans, out)
        return
    for external_id, events in plans:
        _push(external_id, events, force=force)


def _plan_backfill(
    path: Path, arm: str | None, name: str | None = None
) -> list[tuple[str, list[RunEvent]]]:
    """Work out what `path` is and derive every run it holds.

    Args:
        path: A grid directory, a world-model directory, or an optimize manifest.
        arm: One arm of a grid directory, or None for every arm it holds.
        name: Overrides the derived run name (see the command's `--name`).

    Raises:
        typer.BadParameter: The path is neither a grid directory nor an optimize manifest, which is
            almost always a mistyped path rather than an empty directory.
    """
    from wmo.optimize.telemetry.backfill import optimize_events

    if path.is_dir() and (path / COHORT_FILE).is_file():
        return _grid_plans(path, arm, name)
    manifest = path if path.is_file() else path / MANIFEST_RELPATH
    if manifest.is_file():
        model = _manifest_model(manifest)
        # `--name` is the WHOLE run name for a manifest, so it is not put through
        # `pipeline_external_id`: an operator naming a run has to get the name they typed, and
        # appending "/optimize" to it would silently rename it.
        external_id = name or pipeline_external_id(model)
        return [_named(external_id, optimize_events(manifest, model=model))]
    raise typer.BadParameter(
        f"{path} is neither a grid directory (no {COHORT_FILE}) nor an optimize run (no "
        f"{MANIFEST_RELPATH}). Point it at a grid directory under {ARTIFACT_DIR}/, at a built "
        "world model's directory, or at an optimize-run.json."
    )


def _manifest_model(manifest: Path) -> str:
    """Which world model a manifest belongs to, from the manifest itself.

    Read out of the file rather than inferred from its path: the manifest records the world model
    it is a run of, and a run's name must not depend on where someone copied the file to. The
    directory is the fallback for a manifest too old to carry the field.

    Raises:
        typer.BadParameter: The file is not readable JSON, which is a mistyped path far more often
            than it is a damaged artifact.
    """
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        msg = f"{manifest} is not a readable optimize manifest: {error}"
        raise typer.BadParameter(msg) from error
    if isinstance(payload, dict) and isinstance(payload.get("world_model"), str):
        return str(payload["world_model"])
    return manifest.parent.parent.name


def _grid_plans(
    grid_dir: Path, arm: str | None, name: str | None = None
) -> list[tuple[str, list[RunEvent]]]:
    """Every arm of a grid directory, or the one named.

    `name` replaces the grid PREFIX only, and each arm still appends to it, because one grid
    directory holds several arms and each is its own run: a `--name` that named the whole run would
    collapse three arms into one.
    """
    from wmo.optimize.telemetry.backfill import grid_arm_events

    relpath = (name or grid_relpath(grid_dir)).strip("/")
    arms = [arm] if arm is not None else _arms(grid_dir)
    if not arms:
        raise typer.BadParameter(
            f"{grid_dir} holds no arm to replay: no arm directory and no {LEDGER_FILE} naming one. "
            "Name one with --arm if it is somewhere unusual."
        )
    return [
        _named(
            grid_arm_external_id(relpath, arm_name),
            grid_arm_events(grid_dir, arm=arm_name, grid_relpath=relpath),
        )
        for arm_name in arms
    ]


def _named(external_id: str, events: list[RunEvent]) -> tuple[str, list[RunEvent]]:
    """Bind a run's events to the name they will be pushed under.

    The walk stamps each event with the name it DERIVED from the artifacts' location, so a `--name`
    override would otherwise leave the events disagreeing with the run they are pushed to: harmless
    on the wire (the pushed body has no run name, the URL does) but visible in a dry run and exactly
    the kind of divergence that later reads as a bug in the mapping. Rebinding once here keeps one
    answer to "which run is this".
    """
    if all(event.external_id == external_id for event in events):
        return external_id, events
    return external_id, [event.model_copy(update={"external_id": external_id}) for event in events]


def _arms(grid_dir: Path) -> list[str]:
    """The arms a grid directory holds, from its own subdirectories and its ledger.

    Both sources, because either alone misses a case: an arm whose process died before its first
    chunk has a ledger line and no populated directory, and a directory copied without its ledger
    still holds real chunks.
    """
    found = {
        child.name
        for child in grid_dir.iterdir()
        if child.is_dir() and any(child.glob("chunk-*.json"))
    }
    ledger = grid_dir / LEDGER_FILE
    if ledger.is_file():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and isinstance(row.get("arm"), str):
                found.add(str(row["arm"]))
    return sorted(found)


def _write_jsonl(plans: list[tuple[str, list[RunEvent]]], out: Path | None) -> None:
    """Write every planned run's events as JSONL: the dry run, in the shared-fixture format.

    `RunEvent.jsonl_row` is the line shape, which names its run: on the wire the run is in the URL,
    but a file holding several arms has to say which run each line belongs to. Same order and keys
    as the D-RUNS shared-truth fixture, so a dry run is diffable against it line for line.

    Takes ALL the runs at once and truncates once, for two reasons a per-run write got wrong: a
    re-run appended to the previous one and silently doubled the file, and truncating per run would
    have left a multi-arm grid with only its last arm.
    """
    lines = [json.dumps(event.jsonl_row()) for _external_id, events in plans for event in events]
    if out is None:
        for line in lines:
            _console.print_json(line)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    named = ", ".join(external_id for external_id, _events in plans)
    _console.print(f"{len(lines)} event(s) for [bold]{named}[/bold] -> {out}")


def _push(external_id: str, events: list[RunEvent], *, force: bool) -> None:
    """Push one run's events, refusing to double-count a run that already reported itself.

    The seqs come from the artifacts' own walk, so the platform discards any it already holds and a
    repeat is free. What a repeat is NOT free of is the run-level events a LIVE emitter already
    wrote under different seqs, which is why an already-reported run is refused rather than merged:
    its ledger lines would land twice and the spend curve would double.
    """
    from wmo.optimize.telemetry.backfill import BackfillRefused, ensure_backfillable
    from wmo.runtime.platform.client import PlatformError
    from wmo.runtime.runs.client import PushRejected, PushUnavailable, RunsSink, default_emitter_id

    # ONE client for both halves of this command, closed when it is done: the guard reads the run
    # and the push writes it, and opening a second connection pool to do that leaked one per arm.
    with _reader() as reader:
        try:
            ensure_backfillable(reader.event_count(external_id), force=force)
        except BackfillRefused as error:
            raise typer.BadParameter(str(error)) from error
        except PlatformError as error:
            raise _failed(f"Could not check run {external_id!r}", error) from error
        sink = RunsSink(reader.client, org_id=reader.org_id, emitter_id=default_emitter_id())
        try:
            ack = sink.push(external_id, events)
        except (PushRejected, PushUnavailable) as error:
            _console.print(f"[red]Could not push {external_id}:[/red] {escape(str(error))}")
            raise typer.Exit(code=1) from error
    already = len(events) - ack.accepted
    _console.print(
        f"pushed {len(events)} event(s) for [bold]{external_id}[/bold]: {ack.accepted} newly "
        f"accepted" + (f", {already} already recorded (a replay is a no-op)" if already else "")
    )


def _shape_error(what: str, error: ValidationError) -> typer.Exit:
    """Report a response this build cannot read, which is a version skew rather than a crash.

    The platform owns these shapes and can add to them safely, but a REMOVED or renamed field is a
    real mismatch: the remedy is upgrading wmo, and saying that beats a pydantic traceback.
    """
    _console.print(
        f"[red]The platform's answer for {what} is not one this build understands[/red] "
        f"({error.error_count()} field(s)); upgrade wmo, or report the mismatch."
    )
    return typer.Exit(code=1)


def _failed(headline: str, error: PlatformError) -> typer.Exit:
    """Render a failed platform request as a clean error; the message carries the next step."""
    _console.print(f"[red]{headline}:[/red] {escape(str(error))}")
    return typer.Exit(code=1)


def register(app: typer.Typer) -> None:
    """Attach the runs commands to the root CLI."""
    app.add_typer(runs_app, name="runs")


def __getattr__(name: str) -> object:
    """Lazy module attribute resolution for deferred CLI imports."""
    if name == "RunsReader":
        from wmo.runtime.runs.reader import RunsReader

        return RunsReader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

