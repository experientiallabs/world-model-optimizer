"""`wmh eval --mode closed-loop` and `wmh eval agreement` — the closed-loop halves of eval.

Kept out of `app.py` so the (large) eval command stays readable; `app.py` routes here.
Closed-loop mode runs an agent harness against a built world model — the environment is ALWAYS
the world-model simulation; `--harness-backend e2b` only moves the pi-node harness PROCESS into
pooled E2B sandboxes (its tool calls stay answered host-side) — and scores task success;
`agreement` compares two saved closed-loop reports — the outcome-agreement check
docs/reference/closed_loop.md names.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from wmh.cli.model_roles import resolve_opt_in_model_provider
from wmh.config import WorldModelStore
from wmh.engine import load_world_model
from wmh.evals.agreement import compute_agreement
from wmh.evals.closed_loop import ClosedLoopEval, ClosedLoopReport, WorldModelSource
from wmh.evals.gold import GoldJudge, GoldVerdict
from wmh.evals.tasks import load_tasks
from wmh.harness.doc import MAX_TURNS_ID, HarnessDoc, Surface, SurfaceKind
from wmh.harness.runtime import DEFAULT_MAX_TURNS, AgentRuntime
from wmh.harness.store import HarnessStore


def run_closed_loop(
    console: Console,
    *,
    tasks_file: str,
    name: str | None,
    root: str,
    k: int,
    max_turns: int | None,
    out: str | None,
    harness: str | None = None,
    harness_backend: str = "local",
    eval_concurrency: int | None = None,
    e2b_template: str | None = None,
) -> None:
    """Run an agent harness on each task against the world model; print and optionally save.

    `--harness <name>[@ref]` runs a stored harness version (ref = version or alias; default is
    the champion alias); without it the built-in baseline loop runs. `max_turns=None` means "the
    harness's own cap" (or the default for the baseline); an explicit value overrides either —
    never silently ignored. `[models.agent]` selects the agent model for either path; when unset,
    the agent shares the world model's provider. The environment is ALWAYS the world-model
    simulation;
    `--harness-backend e2b` only moves the pi-node harness PROCESS into pooled E2B sandboxes
    (tool calls stay answered by the world model host-side), running all (task, attempt) cells
    at once unless `--eval-concurrency` caps them.
    """
    if harness_backend not in ("local", "e2b"):
        raise typer.BadParameter(
            f"unknown --harness-backend {harness_backend!r}; choose local or e2b"
        )
    try:
        tasks = load_tasks(tasks_file)
    except (OSError, ValueError) as exc:  # missing file, malformed JSONL, empty, duplicate ids
        raise typer.BadParameter(f"cannot load tasks from {tasks_file!r}: {exc}") from exc
    # The world model IS the environment on every backend, so it is always required.
    store = WorldModelStore(root)
    try:
        model_dir = store.resolve(name)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    world_model, provider = load_world_model(model_dir)
    agent_provider, agent_model = resolve_opt_in_model_provider(root, "agent", provider)

    loaded_harness = _load_harness(harness, root)
    if loaded_harness is None and harness_backend == "e2b":
        raise typer.BadParameter(
            "--harness-backend e2b runs a pi-node harness process in sandboxes; the built-in "
            "baseline loop has no such process — pass --harness"
        )
    agent_label = (
        f"{loaded_harness.name}-v{loaded_harness.version}"
        if loaded_harness is not None
        else "baseline"
    )
    agent_identity = (
        f"{agent_provider.config.kind.value}:{agent_model}" if agent_model is not None else None
    )
    report_agent_label = (
        f"{agent_label}[agent={agent_identity}]" if agent_identity is not None else agent_label
    )
    agent_note = (
        f" using agent model [bold]{agent_identity}[/bold]" if agent_identity is not None else ""
    )
    versus = (
        f"world model [bold]{model_dir.name}[/bold]"
        if harness_backend == "local"
        else f"world model [bold]{model_dir.name}[/bold] (pi harness in pooled E2B sandboxes)"
    )
    console.print(
        f"closed-loop: harness [bold]{agent_label}[/bold]{agent_note} vs {versus} "
        f"on {len(tasks)} task(s), k={k}…"
    )

    def _progress(task_id: str, attempt: int, verdict: GoldVerdict) -> None:
        mark = "[green]pass[/green]" if verdict.passed else "[red]fail[/red]"
        console.print(f"  {task_id} #{attempt}: {mark} ({verdict.rationale})")

    if loaded_harness is not None:
        if (
            harness_backend == "local"
            and loaded_harness.runtime_kind() == "pi-node"
            and eval_concurrency is not None
            and eval_concurrency != 1
        ):
            # Local pi runtimes are single-episode resources (one runner port/workdir, or one
            # RunnerLink channel): parallel cells would collide.
            raise typer.BadParameter(
                "pi-node harnesses run one episode at a time under --harness-backend local; "
                "drop --eval-concurrency or use --harness-backend e2b"
            )
        if max_turns is not None and max_turns != loaded_harness.max_turns():
            console.print(
                f"  note: --max-turns {max_turns} overrides the harness's own "
                f"max_turns={loaded_harness.max_turns()}"
            )
            loaded_harness = _with_max_turns(loaded_harness, max_turns)
        try:
            runtime = loaded_harness.runtime(
                agent_provider,
                backend=harness_backend,
                e2b_template=e2b_template,
            )
        except ValueError as exc:  # e.g. e2b on a non-pi-node harness -> usage error
            raise typer.BadParameter(str(exc)) from exc
    else:
        runtime = AgentRuntime(agent_provider, max_turns=max_turns or DEFAULT_MAX_TURNS)
    try:
        evaluation = ClosedLoopEval(
            tasks,
            WorldModelSource(world_model, label=model_dir.name),
            agent_provider,
            GoldJudge(provider),
            label=f"{report_agent_label}@{model_dir.name}",
            k=k,
            concurrency=(
                eval_concurrency
                if eval_concurrency is not None
                else (0 if harness_backend == "e2b" else 1)
            ),
            runtime=runtime,
            on_progress=_progress,
        )
        report = evaluation.run()
    finally:
        if harness_backend == "e2b":
            # An eval-owned e2b runtime owns a private sandbox pool; tear it down with the eval.
            from wmh.harness.pi_e2b import E2BPiRuntime

            if isinstance(runtime, E2BPiRuntime):
                runtime.close()
    for task_id, outcome in report.per_task.items():
        console.print(
            f"  {task_id:24} success={outcome.success_rate:.2f} "
            f"assertions={outcome.mean_fraction:.2f}"
        )
    console.print(f"[bold]OVERALL[/bold] {report.summary()}")
    if out:
        Path(out).write_text(report.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"wrote closed-loop report -> {out}")


def run_agreement(console: Console, *, report_a: str, report_b: str, threshold: float) -> None:
    """Compare two saved closed-loop reports task-by-task and print the agreement verdict."""
    a = _load_report(report_a)
    b = _load_report(report_b)
    result = compute_agreement(a, b, pass_threshold=threshold)
    c = result.confusion
    la, lb = result.label_a or "A", result.label_b or "B"
    console.print(f"[bold]task verdict confusion[/bold] ({la} vs {lb}):")
    console.print(f"  {la}-pass & {lb}-pass: {c.a_pass_b_pass}")
    console.print(f"  {la}-pass & {lb}-FAIL: {c.a_pass_b_fail}  (A over-credits these)")
    console.print(f"  {la}-FAIL & {lb}-pass: {c.a_fail_b_pass}")
    console.print(f"  {la}-FAIL & {lb}-FAIL: {c.a_fail_b_fail}")
    console.print(f"[bold]VERDICT[/bold] {result.summary()}")


def _load_report(path: str) -> ClosedLoopReport:
    try:
        return ClosedLoopReport.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(f"cannot read closed-loop report {path!r}: {exc}") from exc


def _with_max_turns(doc: HarnessDoc, max_turns: int) -> HarnessDoc:
    """A copy of `doc` with its max-turns surface replaced (re-validated via the constructor)."""
    surfaces = [s for s in doc.surfaces if s.id != MAX_TURNS_ID]
    surfaces.append(Surface(id=MAX_TURNS_ID, kind=SurfaceKind.PARAM, content=str(max_turns)))
    return HarnessDoc(name=doc.name, version=doc.version, surfaces=surfaces)


def _load_harness(name: str | None, root: str) -> HarnessDoc | None:
    if name is None:
        return None
    base, _, ref = name.partition("@")
    try:
        return HarnessStore(root).load(base, ref or None)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
