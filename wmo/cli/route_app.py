"""`wmo optimize route`: sweep, fit, tune, and report learned inference policies.

The routing optimizer's CLI face, sitting beside `wmo optimize harness` in the optimizer
family. The workflow chains in one direction:

    student -> pool -> sweep -> OutcomeMatrix -> fit -> policy.json -> tune / report

`student` puts a freshly distilled adapter into the candidate pool, which is what makes a trained
model routable at all. `sweep` is the producer: it measures every candidate on the world model's
own held-out scenarios and writes the `OutcomeMatrix` everything downstream consumes (a research
adapter such as RouterBench can write the same artifact instead). `fit` emits the policy artifact
serving loads, `report` the improvement report the endpoint cites, and `tune` is the one post-fit
control: it moves a fitted policy's cost/quality dial without refitting.

`pin` sits outside that chain: it installs a `kind="static"` policy for one pool model, so a
single candidate is serveable before any measurement exists, which is the honest zero-evidence
starting point a fit is compared against. Vocabulary note: "route" is developer-facing CLI only;
customer copy never says router.
"""

from __future__ import annotations

import itertools
import json
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape
from rich.prompt import Confirm
from rich.table import Table

from wmo.cli.consent import require_spend_consent
from wmo.config import ARTIFACT_DIR, WorldModelStore

if TYPE_CHECKING:
    # Type-only: real imports are local to the commands and helpers that construct or inspect
    # these values, so importing this module never pulls the optimize/engine/env/distill/pool
    # bodies behind it.
    from llm_waterfall import ChatMaxTokensField

    from wmo.optimize.knn import DialResult, KnnFitOutcome
    from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
    from wmo.optimize.policy import RoutingPolicy
    from wmo.optimize.sweep import CandidateCoverage, DeferredRisk, SweepPlan, SweepRun
    from wmo.providers.pool import PoolEntry

# The two output-budget parameter names any OpenAI-compatible backend accepts.
_MAX_TOKENS_FIELDS: tuple[ChatMaxTokensField, ...] = ("max_tokens", "max_completion_tokens")

route_app = typer.Typer(
    help="Make models routable, measure them closed-loop, then fit, tune, and report policies.",
    no_args_is_help=True,
)

_console = Console()

DEFAULT_MATRIX_FILENAME = "matrix.json"
"""Default `sweep --out`: the outcome matrix `fit` takes as its argument."""

# Literal mirrors of constants that otherwise live behind a heavy import
# (`wmo.optimize.compression`, `wmo.optimize.policy`, `wmo.env.llm_agent`, `wmo.providers.pool`).
# Typer evaluates Option defaults and f-string help text at command-definition time, so these
# have to be values, not names imported from those modules; the real constants are re-imported
# inside the command bodies that need their behavior.
_DEFAULT_HISTORY_CHARS = 2000
_DEFAULT_POOL_PATH = ".wmo/pool.toml"
_POLICY_FILENAME = "policy.json"
_HASHING_EMBEDDER_DIM = 512
_AZURE_EMBEDDER_DIM = 3072
_AZURE_EMBEDDER_ENV = ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT")
_WM_SIMULATED = "wm_simulated"
_REAL_EPISODE = "real_episode"
_DEFAULT_WM_JUDGE = "world-model verifier"

COMPRESSOR_IDS_HELP = "identity | llmlingua2-endpoint | truncate"
"""What `--compressor` accepts. Mirrors `wmo.optimize.compression.registered_compressor_ids()`."""

_MATRIX_DIGEST_MARK = "sha256="
"""How `load_matrix_with_digest` spells the digest inside a policy's `fitted_from`."""


@route_app.command("sweep")
def sweep(
    model: str = typer.Argument(
        None, help="World model to measure against (default: the only one built under --root)."
    ),
    pool_file: str = typer.Option(
        _DEFAULT_POOL_PATH,
        "--pool",
        # The doubled brackets are escaped: typer renders help through rich markup, which
        # otherwise swallows them and prints an empty pair.
        help="Candidate pool TOML: one \\[\\[model]] table per candidate.",
    ),
    traces_file: str = typer.Option(
        None,
        "--traces",
        help="Trace corpus the scenarios come from (default: the model's own "
        "traces.otel.jsonl, as `wmo demo --traces` resolves it). A build does not keep a copy "
        "of the corpus it read, so pass the file here.",
    ),
    scenarios: int = typer.Option(
        20,
        "--scenarios",
        min=1,
        help="Cap on held-out scenarios measured (a deterministic prefix by trace id).",
    ),
    episodes: int = typer.Option(
        1, "--episodes", min=1, help="Episodes per (candidate, scenario) cell."
    ),
    max_steps: int = typer.Option(
        20, "--max-steps", min=1, help="Step budget per episode (also the cost estimate's cap)."
    ),
    concurrency: int = typer.Option(
        1,
        "--concurrency",
        min=1,
        help="Cells measured at once (1 = one at a time). Changes only how long the sweep takes, "
        "never what it measures, and a sweep interrupted at one value resumes at another. Your "
        "PROVIDER LIMITS are the real ceiling, not this number: every candidate call and every "
        "world-model serve and judge call is a request, and the world model's own calls all come "
        "out of ONE account's bucket, so raising this past what that bucket allows turns cells "
        "into throttling errors instead of results.",
    ),
    history_chars: int = typer.Option(
        _DEFAULT_HISTORY_CHARS,
        "--history-chars",
        min=1,
        help="Characters of each observation the agent sees on later turns. Raise it for an "
        "environment whose tool payloads are large: too small and the agent cannot see what it "
        "just fetched, so it re-fetches. Changes what candidates are measured on, so matrices "
        "swept at different values are not comparable.",
    ),
    assume_input_tokens: int = typer.Option(
        2000,
        "--assume-input-tokens",
        min=0,
        help="ASSUMED input tokens per policy call, for the cost estimate only.",
    ),
    assume_output_tokens: int = typer.Option(
        250,
        "--assume-output-tokens",
        min=0,
        help="ASSUMED output tokens per policy call, for the cost estimate only.",
    ),
    out: str = typer.Option(
        DEFAULT_MATRIX_FILENAME, "--out", help="Where to write the OutcomeMatrix JSON."
    ),
    root: str = typer.Option(ARTIFACT_DIR, "--root", help="Project dir."),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Consent to the projected spend up front. Required in a non-interactive "
        "session (CI, cron, piped output, redirected input), where the run otherwise "
        "refuses to start.",
    ),
    allow_uneven_coverage: bool = typer.Option(
        False,
        "--allow-uneven-coverage",
        help="Hand the matrix to `fit` even when the candidates were not scored on the same "
        "evidence: different scenarios, or different numbers of surviving episodes on the same "
        "scenarios. The fit is then biased (both fitters skip unscored rows and weigh the rest per "
        "episode); the coverage table prints either way.",
    ),
    compressor: str = typer.Option(
        None,
        "--compressor",
        help="D-COMPRESS: measure every candidate call through this compressor "
        f"({COMPRESSOR_IDS_HELP}), so the matrix is the compressed ARM of the grid. Default: "
        "uncompressed. `fit` requires the matrix arm to match the policy it stamps.",
    ),
    aggressiveness: float = typer.Option(
        0.0,
        "--aggressiveness",
        min=0.0,
        max=1.0,
        help="Compressor-defined dial in [0, 1] for --compressor: 0.0 is a no-op and higher "
        "never removes less, but it is not an exact removal fraction (the achieved ratio is "
        "recorded per episode).",
    ),
) -> None:
    """Measure every pool candidate closed-loop and write the outcome matrix `fit` consumes.

    This is step one of the routing workflow: nothing else produces an `OutcomeMatrix`.

        wmo optimize route sweep support --traces traces.otel.jsonl --pool .wmo/pool.toml
        wmo optimize route fit matrix.json --kind knn

    Every (candidate, scenario, episode) cell runs one full episode against the world model,
    which scores it (`WorldModelEnv(..., score_on_close=True)`): a matrix without verified
    rewards is not evidence. Scenarios are the task prompts of the corpus's TEST band, the third
    band of the build's deterministic 3-way split, which prompt optimization and knowledge
    extraction never saw; the candidates' tool surface is summarized from the TRAIN band only
    (the same discipline), so a candidate is not scored on guessing what tools exist. What that
    buys is a policy fitted on prompts no GEPA candidate was SELECTED on. It is not isolation
    from the environment: a build indexes the full corpus for serving, so the world model can
    still retrieve a held-out trace's own recorded steps as demos when it simulates that
    scenario. Bands are also cut per trace id, not per task text, so a task repeated across
    traces can appear on both sides. The whole sweep runs the world model frozen, so no cell's
    predictions become another cell's retrieved demos and the result does not depend on sweep
    order.

    Nothing measured is lost, and nothing measured is bought twice. Every cell lands in
    `<out>.partial.jsonl` the moment it completes, so a sweep killed at hour five keeps the cells
    it paid for; re-running the same command measures only what is missing and then writes the
    matrix and removes the sidecar. Changing what the sweep measures (the pool, the scenario cut,
    episodes, the step budget, the observation window, the compressor) makes those rows a
    different arm, and the command says so and stops rather than merging two arms into one matrix.
    `--concurrency N` runs N cells at once, which is the difference between a six-hour grid and a
    one-hour grid; it is not part of what the sweep measures, so a run interrupted at one value
    resumes at another.

    Spend is confirmed before the first episode runs, and consent is said, never inferred: at a
    terminal the projected cost is a question, and with nobody to ask (CI, cron, piped output,
    `| tee`, or a redirected stdin, which is not a person even when stdout is a terminal) the run
    REFUSES with exit code 2 unless `--yes` was passed, naming what it would have spent.
    What that estimate multiplies is ASSUMED tokens per policy call by the real
    cell and call counts, so it is a projection, never a measurement; the measured candidate spend
    is printed when the sweep finishes. Before that question is asked, every candidate's backend
    is resolved as far as it goes without a request: its kind's static
    requirements from the entry alone, then its lazy SDK client forced to BUILD, which imports the
    SDK and resolves credentials locally. So a candidate that could never be called is a usage
    error at the boundary, not a mid-sweep abort with earlier candidates already paid for. Two
    things stay first-cell failures because seeing them needs a request (bedrock AWS credentials,
    tinker service reachability); the pre-flight names them per entry when the pool has one.

    Fit-readiness is a coverage contract, not a nonzero count. A cell goes unscored when its
    episode errored (provider throttle, agent crash, judge failure), both fitters SKIP unscored
    rows, and what they do with the rest is episode-weighted, so the contract is that every
    candidate has the same number of scored episodes on the same scenarios. Two ways to break it,
    both blocked: a matrix where candidate A was scored on 20 scenarios and B on 11 ranks them on
    DIFFERENT task sets, and a matrix where both cover all 20 but A kept 3 episodes on a scenario
    where B kept 1 weighs that scenario three times as heavily for A, because `--kind rank`
    averages every surviving episode into its cluster mean and both kinds pick their
    default/fallback model off episode-weighted means (`routing._overall_best`,
    `knn.best_single_on_fit`). A knn BANK cell is that pair's own mean, so the bank is milder, but
    milder is not unbiased and it is the same matrix either way. Either break leaves the policy
    decided by whichever cells each candidate happened to lose. So per-candidate scored counts
    ALWAYS print, and when the evidence differs the command still writes the matrix (those cells
    were paid for, and their `error` fields are the diagnosis) but WITHHOLDS the `fit` handoff and
    exits non-zero, naming each candidate, the scenarios it has no scored episode for, and the
    scenarios where it kept fewer episodes than the best-covered candidate.
    `--allow-uneven-coverage` is the opt-out for an operator who knows the bias and wants the
    partial data anyway (one candidate's backend down for the whole sweep, say): it prints the same
    coverage table and stops treating the difference as fatal. Losing the SAME cells for every
    candidate is not uneven, since the comparison stays like-for-like on less data and the counts
    show the loss.

    Exit code 1 when the matrix is not fit-ready: no cell scored at all, or unequal scored evidence
    without `--allow-uneven-coverage`. `sweep && fit` in a script then stops instead of fitting on
    it, and the matrix is written either way.
    """
    from wmo.engine import load_world_model
    from wmo.env import WorldModelEnv
    from wmo.optimize.compression import resolve_compression
    from wmo.optimize.sweep import (
        SweepError,
        coverage,
        execute_sweep,
        plan_sweep,
        preflight_pool,
        resolve_config,
        resumable_cells,
    )

    out_path = Path(out)
    if compressor is None and aggressiveness > 0.0:
        raise typer.BadParameter("--aggressiveness needs --compressor to apply it")
    sweep_compression = None
    if compressor is not None:
        try:
            # Checked before a single episode is paid for, and against the SERVING rule: there
            # is no point measuring an arm whose compressor could never be mounted.
            sweep_compression = resolve_compression(compressor, aggressiveness)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    store = WorldModelStore(root)
    try:
        model_dir = store.resolve(model)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except ValueError as exc:
        # `resolve` says "pass --name", the option `wmo serve`/`play`/`demo` carry. Here the
        # model is a POSITIONAL, so say what a user of this command actually types.
        names = store.list_names()
        raise typer.BadParameter(
            f"multiple world models built ({', '.join(names)}); name one as the MODEL argument, "
            f"e.g. `wmo optimize route sweep {names[0]}`"
        ) from exc
    # Everything knowable without spending is settled BEFORE the cost question: a candidate whose
    # backend cannot even be constructed, or an --out that cannot be written, would otherwise
    # surface after the sweep had already paid for cells it then throws away.
    try:
        config = resolve_config(model_dir)
        preflight = preflight_pool(Path(pool_file))
    except SweepError as exc:
        raise typer.BadParameter(str(exc)) from exc
    print_deferred_risks(_console, preflight.deferred)
    try:
        plan = plan_sweep(
            model_dir=model_dir,
            config=config,
            pool=preflight.pool,
            out_path=out_path,
            traces_file=Path(traces_file) if traces_file is not None else None,
            scenarios=scenarios,
            episodes=episodes,
            max_steps=max_steps,
            assume_input_tokens=assume_input_tokens,
            assume_output_tokens=assume_output_tokens,
            history_chars=history_chars,
            compression=sweep_compression,
            max_concurrency=concurrency,
        )
    except SweepError as exc:
        raise typer.BadParameter(str(exc)) from exc
    print_tiny_corpus_note(_console, plan)
    try:
        # Before the money question, not after it: a sidecar left by a run of a DIFFERENT plan is
        # refused here, while refusing still costs nothing.
        already_measured = resumable_cells(plan)
    except SweepError as exc:
        raise typer.BadParameter(str(exc)) from exc
    world_model, _serve_provider = load_world_model(model_dir)

    print_cost_estimate(_console, plan, already_measured=already_measured)
    _confirm_cost(plan, yes=yes)

    _console.print(
        f"sweeping {len(plan.pool.models)} candidate(s) over {len(plan.scenarios)} held-out "
        f"scenario(s) of [bold]{escape(model_dir.name)}[/bold], {episodes} episode(s) each…"
    )
    run = execute_sweep(
        plan,
        world_model=world_model,
        env_factory=lambda: WorldModelEnv(world_model, score_on_close=True),
        on_outcome=cell_progress(_console, plan.cells - already_measured),
    )
    matrix = run.matrix
    scored = sum(1 for outcome in matrix.outcomes if outcome.scored)
    # `escape(out)`: a bracketed path segment would otherwise be read as markup and dropped, so
    # the line would print a path that does not exist (and this one is meant to be copied).
    _console.print(
        f"[green]✓[/green] {len(matrix.outcomes)} cell(s), {scored} scored -> {escape(out)}\n"
        f"  measured candidate spend ${run.candidate_usd:.4f}{_compressor_note(run)} (the world "
        "model's own serve/judge cost is metered separately)",
        soft_wrap=True,  # a path a user copies must not be wrapped
    )
    print_world_model_spend(_console, run)
    rows = coverage(matrix)
    print_coverage(_console, rows)
    if scored == 0:
        # Exit non-zero: a matrix with no verified reward is not evidence, so `sweep && fit` in a
        # script must stop here rather than fit on it. The rows are on disk for their `error`s.
        # No --allow-uneven-coverage escape: there is nothing to fit, so `fit` would fail anyway.
        _console.print(NO_EVIDENCE_WARNING)
        raise typer.Exit(1)
    warning = uneven_warning(rows)
    if warning is not None:
        _console.print(warning)
        if not allow_uneven_coverage:
            _console.print(
                "  fix the lost cells and sweep again, drop the candidate that lost them, or "
                "re-run with [bold]--allow-uneven-coverage[/bold] to fit on this matrix anyway"
            )
            raise typer.Exit(1)
        _console.print(BIAS_ACCEPTED_NOTE)
    _console.print(
        f"  next: [bold]wmo optimize route fit {escape(out)} --kind knn[/bold]", soft_wrap=True
    )


# ------------------------------------------------------------------ shared sweep presentation
# Rendering for the sweep's typed results, shared by `route sweep` and `optimize model`'s sweep
# stage so the coverage contract reads the same whichever command a user reached it through.
# Every one takes its console explicitly: the two commands own different ones.

NO_EVIDENCE_WARNING = (
    "[yellow]warning[/yellow] no cell was scored, so this matrix is not evidence and "
    "fitting will fail; read the `error` field of a row to see what broke"
)

BIAS_ACCEPTED_NOTE = (
    "  --allow-uneven-coverage was passed: fitting on it anyway, with that bias accepted"
)

# Scenario ids shown per candidate before the column summarizes the rest: enough to see the pattern
# in a table an operator reads, without a 20-id line per row.
_LOST_SHOWN = 5


def print_deferred_risks(console: Console, deferred: tuple[DeferredRisk, ...]) -> None:
    """Name what the request-free pre-flight could not close, per candidate that carries it."""
    if not deferred:
        return
    console.print(
        "[yellow]note[/yellow] the pre-flight makes no request, so one thing per candidate "
        "below can still fail at its first cell (the matrix records it as that cell's `error`):"
    )
    for risk in deferred:
        console.print(f"  {escape(risk.candidate)} (kind={risk.kind.value}): {risk.risk}")


def print_tiny_corpus_note(console: Console, plan: SweepPlan) -> None:
    """Say when the corpus was too small to leave a held-out band to measure on."""
    if not plan.tiny_corpus:
        return
    console.print(
        f"[yellow]note[/yellow] {plan.trace_count} trace(s) is too few for a held-out band, so "
        "these scenarios come from the FULL corpus: they are not leak-free, and a policy "
        "fitted on them is a smoke test, not evidence"
    )


def _compressor_note(run: SweepRun) -> str:
    """Name the compressor's share of candidate spend, but only when one actually billed.

    The D-COMPRESS rule folds the compressor's inference cost into the candidate figure, so on a
    compressed arm that number is not just the models. Saying so on an UNCOMPRESSED sweep would
    be noise about a stage that did not run, which is why this is conditional on a nonzero bill
    rather than on the flag.
    """
    if run.compressor_usd <= 0.0:
        return ""
    return f" (incl. ${run.compressor_usd:.4f} compressor)"


def print_world_model_spend(console: Console, run: SweepRun) -> None:
    """The OTHER half of a sweep's bill: what the simulator charged to run the evaluation.

    Printed as its own line, never folded into the candidate figure above it. The candidate side
    is the serving cost a customer would pay and the policy is fitted to trade off; this side is
    eval infrastructure that exists only because the measurement happened. One number covering
    both would misprice both.
    """
    gap = run.metering_gap
    if run.episodes_metered == 0:
        console.print(f"  world-model spend {gap}")
        return
    usage = run.world_model_usage
    phases = ", ".join(
        f"{phase.value} ${bucket.cost_usd:.4f}" for phase, bucket in sorted(usage.by_phase.items())
    )
    detail = f" ({phases})" if phases else ""
    console.print(
        f"  measured world-model spend ${run.world_model_usd:.4f} over {run.episodes_metered} "
        f"session(s){detail}: eval infrastructure, not serving cost"
        + (f"\n  [yellow]note[/yellow] {gap}" if gap is not None else ""),
        soft_wrap=True,
    )
    if run.usage_path is not None:
        console.print(
            f'  recorded as kind="sweep" -> {escape(str(run.usage_path))}', soft_wrap=True
        )


def cell_progress(console: Console, cells: int) -> Callable[[ScenarioOutcome], None]:
    """A per-cell progress line: which cell, what it scored, what it cost."""
    done = itertools.count(1)

    def _on_outcome(outcome: ScenarioOutcome) -> None:
        reward = "unscored" if outcome.reward is None else f"{outcome.reward:.2f}"
        console.print(
            f"  [{next(done)}/{cells}] {escape(outcome.model)} {escape(outcome.scenario_id)} "
            f"ep{outcome.episode}: reward={reward} ${outcome.cost_usd:.5f} "
            f"steps={outcome.steps}"
        )

    return _on_outcome


def uneven_warning(rows: list[CandidateCoverage]) -> str | None:
    """The warning for coverage that is not a comparison, or None when it is one.

    Two different failures, so two different messages: candidates ranked on different scenario
    SETS, and candidates ranked on the same scenarios with different numbers of surviving EPISODES.
    Both bias a fit; naming which one happened is what makes the message actionable.
    """
    from wmo.optimize.sweep import Unevenness, unevenness

    counts = ", ".join(f"{escape(row.candidate)} {row.scored}" for row in rows)
    match unevenness(rows):
        case Unevenness.EVEN:
            return None
        case Unevenness.SCENARIOS:
            return (
                "[yellow]warning[/yellow] candidates were scored on DIFFERENT scenarios (scored "
                f"cells: {counts}), so `fit` would rank them on different task sets: it skips "
                "unscored rows, and the policy that comes out is biased by which scenarios each "
                "candidate lost. The paid cells are on disk and their `error` field says what "
                "broke."
            )
        case Unevenness.EPISODES:
            return (
                "[yellow]warning[/yellow] candidates cover the same scenarios but kept DIFFERENT "
                f"numbers of scored episodes on them (scored cells: {counts}; the table above "
                "says which scenarios were thinned, as kept/most). Both fitters weigh EPISODES: "
                "--kind rank averages every surviving episode into its cluster mean, so a "
                "scenario one candidate kept 1 of 3 episodes on counts a third as much for it, "
                "and both kinds pick their default/fallback model off the same episode-weighted "
                "means. What comes out then turns on which episodes happened to fail."
            )


def print_coverage(console: Console, rows: list[CandidateCoverage]) -> None:
    """Show what each candidate would be weighed on: its scored cells, and what it lost.

    The last column names every scenario where this candidate holds less evidence than the
    best-covered candidate does: a bare id is a scenario with no scored episode at all, and
    `id 1/3` is a scenario where it kept 1 of the 3 episodes another candidate kept. Both change
    what a fitter weighs, so both are per candidate here rather than summed into Unscored.
    """
    most: Counter[str] = Counter()
    for row in rows:
        for sid, count in row.scored_episodes:
            most[sid] = max(most[sid], count)
    table = Table(title="Scored coverage per candidate (`fit` SKIPS unscored cells)")
    table.add_column("Candidate", no_wrap=True)
    table.add_column("Scored", justify="right")
    table.add_column("Unscored", justify="right")
    table.add_column("Scenarios lost, or thinned (kept/most)")
    for row in rows:
        gaps = [
            sid if count == 0 else f"{sid} {count}/{most[sid]}"
            for sid, count in row.scored_episodes
            if count == 0 or count < most[sid]
        ]
        lost = ", ".join(gaps[:_LOST_SHOWN])
        if len(gaps) > _LOST_SHOWN:
            lost += f" (+{len(gaps) - _LOST_SHOWN} more)"
        # Scenario ids are corpus data (trace ids) and candidate names are operator strings: both
        # reach a rich console, where `[a]` is markup that would silently drop from the table.
        table.add_row(
            escape(row.candidate),
            f"{row.scored:,}",
            f"{row.unscored:,}",
            escape(lost) if lost else "-",
        )
    console.print(table)
    for row in rows:
        if row.scored == 0 and row.first_error is not None:
            # A candidate that was never scored at all is the one failure a coverage table cannot
            # explain, and its cause is already on disk. Surfacing the first one names the entry
            # and points at the pool file, which is where the fix is.
            console.print(
                f"  [yellow]{escape(row.candidate)}[/yellow] was never scored; its first cell "
                f"failed with: {escape(row.first_error)}\n"
                "    fix that entry in the pool file, drop it, or retry once the cause has cleared"
            )


def print_cost_estimate(console: Console, plan: SweepPlan, *, already_measured: int = 0) -> None:
    """Render the projected spend, stating exactly which parts are assumed.

    Honest by construction: the CELL and CALL counts are real (the step budget is the per-episode
    cap, so calls are an upper bound), the tokens per call are an assumption the flags name, and
    the per-candidate $/Mtok is the pool entry's own price row. The world model's serve and judge
    calls are a separate meter (the D12 cost split) and are deliberately absent.

    `already_measured` is the count a resume will reuse instead of buying. The table above it is
    still the whole grid, because that is what the per-candidate arithmetic describes; the line
    under it says how much of that grid this run is actually paying for.
    """
    table = Table(title="Route sweep cost estimate (ASSUMED tokens, not a measurement)")
    table.add_column("Candidate", no_wrap=True)
    table.add_column("Episodes", justify="right")
    table.add_column("Calls (max)", justify="right")
    table.add_column("$/Mtok in", justify="right")
    table.add_column("$/Mtok out", justify="right")
    table.add_column("USD (est)", justify="right")
    for line in plan.cost_lines:
        table.add_row(
            # A rich cell renders markup: an operator-chosen name like `gpt[a]` would print as
            # `gpt`, making two candidates indistinguishable in the table they confirm spend from
            # (and a name containing a closing tag would abort the command outright).
            escape(line.candidate),
            f"{line.episodes:,}",
            f"{line.calls:,}",
            f"{line.input_per_mtok:.3f}",
            f"{line.output_per_mtok:.3f}",
            f"{line.usd:.2f}",
        )
    console.print(table)
    console.print(
        f"{plan.cells} cell(s) = {len(plan.cost_lines)} candidate(s) x "
        f"{len(plan.scenarios)} held-out scenario(s) x {plan.episodes} episode(s); estimated "
        f"total ${plan.total_usd:.2f}"
    )
    if already_measured:
        console.print(
            f"  RESUMING: {already_measured} of those cell(s) are already measured beside the "
            f"matrix and are NOT bought again, so this run measures {plan.cells - already_measured}"
            " and spends proportionally less than the total above."
        )
    if plan.max_concurrency > 1:
        console.print(
            f"  {plan.max_concurrency} cell(s) run at once, so the sweep finishes sooner for the "
            "same money; it does not change what is measured."
        )
    console.print(
        f"  ASSUMPTION: {plan.assume_input_tokens:,} input + {plan.assume_output_tokens:,} output "
        f"token(s) per policy call, and every episode running its full {plan.max_steps}-call "
        "budget. Calls are capped, tokens per call are NOT measured: set "
        "--assume-input-tokens/--assume-output-tokens from your own numbers, or read the measured "
        "spend this command prints when it finishes."
    )
    console.print(
        "  Candidate side only: the world model's own serve and judge calls are metered "
        "separately and are NOT in this figure."
    )


def _confirm_cost(plan: SweepPlan, *, yes: bool) -> None:
    """Confirm the projected spend before any episode runs.

    Consent is said, never inferred: a non-interactive session cannot answer a prompt, so a
    spending run REFUSES unless `--yes` was passed. This command shipped proceed-and-note for
    its first day, and the equivalent branch in `wmo optimize model` spent a scripted caller's
    real money it never agreed to; every spend surface now shares one refusal
    (`wmo.cli.consent.require_spend_consent`).

    Raises:
        typer.Exit: The user declined (exit code 0), or a non-interactive session was not told
            `--yes` (exit code 2).
    """
    if not require_spend_consent(
        _console,
        yes=yes,
        spend=f"~${plan.total_usd:.2f} across {plan.cells} cell(s)",
        command="wmo optimize route sweep",
    ):
        raise typer.Exit(0)


@route_app.command("student")
def student(
    card_dir: str = typer.Argument(
        ...,
        help="The distillation run dir, or an adapter version dir: whichever holds "
        "model_card.json.",
    ),
    input_per_mtok: float = typer.Option(
        ...,
        "--input-per-mtok",
        min=0.0,
        help="Prompt-token price at the serving endpoint, USD per 1M tokens. Required: an "
        "unpriced candidate reports $0 and a cost-aware policy would route everything to it.",
    ),
    output_per_mtok: float = typer.Option(
        ...,
        "--output-per-mtok",
        min=0.0,
        help="Completion-token price at the serving endpoint, USD per 1M tokens.",
    ),
    name: str = typer.Option(
        "student", "--name", help="Pool handle: what policy artifacts and request logs call it."
    ),
    pool: str = typer.Option(
        _DEFAULT_POOL_PATH, "--pool", help="Candidate pool TOML to add the entry to."
    ),
    endpoint: str = typer.Option(
        None,
        "--endpoint",
        help="OpenAI-compatible base URL. Default: Tinker's serving endpoint.",
    ),
    api_key_env: str = typer.Option(
        None,
        "--api-key-env",
        help="Env var holding the endpoint's API key. Default: TINKER_API_KEY on Tinker's own "
        "endpoint; on any other --endpoint the provider's WMO_ENDPOINT_API_KEY fallback is used, "
        "so a Tinker key is never sent to a host you named.",
    ),
    chat_max_tokens_field: str = typer.Option(
        None,
        "--chat-max-tokens-field",
        help="Output-budget parameter the endpoint accepts: max_tokens | max_completion_tokens. "
        "Default: max_tokens on Tinker's endpoint, max_completion_tokens on any other.",
    ),
    yes: bool = typer.Option(
        False, "--yes", help="Skip the confirmation when an entry of this name already exists."
    ),
) -> None:
    """Add a distilled student to the candidate pool, so the router can select it.

    The keystone step between training and serving: a run produces a `tinker://` adapter, and this
    turns it into a `\\[\\[model]]` entry the sweep measures, the fitter routes to, and the endpoint
    calls, with no hand-edited TOML in between:

        wmo optimize route student .wmo/distill/support --input-per-mtok 0.1 --output-per-mtok 0.4

    On Tinker's own endpoint the entry reads its credential from `TINKER_API_KEY`, so export that
    before serving. Point `--endpoint` somewhere else and the Tinker defaults do NOT follow: the
    entry falls back to `WMO_ENDPOINT_API_KEY` and to `max_completion_tokens`, so a Tinker key is
    never sent to a host you named. `--api-key-env` and `--chat-max-tokens-field` set either
    explicitly.

    To serve the student on its own with no measurement at all, follow this with
    `wmo optimize route pin <world-model> --model student`; to have the router CHOOSE between the
    student and the rest of the roster, run `wmo optimize route fit` on a matrix that covers both.
    """
    from wmo.core.locks import FileLockTimeout
    from wmo.distill.store import MODEL_CARD_FILE, DistillModelCard, student_pool_entry
    from wmo.providers.pool import upsert_pool_entry

    card_path = Path(card_dir) / MODEL_CARD_FILE
    if not card_path.is_file():
        raise typer.BadParameter(
            f"no {MODEL_CARD_FILE} at {card_path}; pass a distillation run directory (the one "
            "holding config.toml and metrics.jsonl) or an adapter version directory "
            "(.wmo/adapters/<name>/vN)"
        )
    if endpoint is not None and not endpoint.strip():
        # `--endpoint "$UNSET_VAR"` is the way this happens. Falling back to Tinker's endpoint
        # would silently serve a different host than the script meant to name.
        raise typer.BadParameter(
            "--endpoint is empty; give the OpenAI-compatible base URL, or drop the flag to use "
            "Tinker's serving endpoint"
        )
    if api_key_env is not None and not api_key_env.strip():
        # Same accident as an empty --endpoint. An empty string reaches `pool_provider` as a
        # falsy api_key_env, which it reads as "no explicit credential" and skips its own
        # unset-variable check, so the misconfiguration would only surface as a 401 at request
        # time with no hint.
        raise typer.BadParameter(
            "--api-key-env is empty; name the environment variable holding the endpoint's key, "
            "or drop the flag to use the provider's default credentials"
        )
    if chat_max_tokens_field is not None and chat_max_tokens_field not in _MAX_TOKENS_FIELDS:
        raise typer.BadParameter(
            f"unknown --chat-max-tokens-field {chat_max_tokens_field!r}; use "
            f"{' or '.join(_MAX_TOKENS_FIELDS)}"
        )
    try:
        card = DistillModelCard.model_validate_json(card_path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise typer.BadParameter(f"cannot read the model card at {card_path}: {exc}") from exc
    try:
        entry = student_pool_entry(
            card,
            name=name,
            input_per_mtok=input_per_mtok,
            output_per_mtok=output_per_mtok,
            endpoint=endpoint,
            api_key_env=api_key_env,
            chat_max_tokens_field=cast("ChatMaxTokensField | None", chat_max_tokens_field),
        )
    except ValidationError as exc:
        raise typer.BadParameter(f"cannot build a pool entry for '{name}': {exc}") from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    pool_path = Path(pool)
    if _pool_has(pool_path, name) and not yes and not _confirm_replace(pool_path, name):
        _console.print(
            f"left {pool_path} unchanged; pass --yes to replace '{name}' (comments in the file "
            "are not preserved by a replacement), or --name <other> to keep both"
        )
        raise typer.Exit(0)
    if _pool_disabled(pool_path, name):
        # Same rule as the registry writer: `enabled = false` is an explicit operator edit,
        # and replacing the entry must not silently put the candidate back into selection.
        entry = entry.model_copy(update={"enabled": False})
        _console.print(
            f"[dim]'{name}' is disabled in the roster (enabled = false); keeping it disabled[/dim]"
        )
    try:
        written = upsert_pool_entry(entry, pool_path)
    except FileLockTimeout as exc:
        # Nothing is wrong with the flags, so this is not a BadParameter: another writer is in the
        # way. Exit non-zero (and say to retry) so a script does not read it as a registration.
        _console.print(f"[red]pool busy[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    verb = "replaced" if written.replaced else "added"
    rewrite_note = (
        "\n  the roster was rewritten, so its comments are gone" if written.rewritten else ""
    )
    _console.print(
        f"[green]✓[/green] {verb} pool candidate [bold]{name}[/bold]{rewrite_note} -> {pool_path}\n"
        f"  {card.base_model} adapter at {entry.model}\n"
        f"  ${input_per_mtok:g}/${output_per_mtok:g} per 1M in/out tokens, "
        f"{_credential_note(entry)}\n"
        f"  serve it directly: wmo optimize route pin <world-model> --model {name}",
        soft_wrap=True,
    )


def _credential_note(entry: PoolEntry) -> str:
    """How this entry authenticates, so the summary never names a key it will not send."""
    if entry.api_key_env is not None:
        return f"credential from {entry.api_key_env}"
    return "credential from WMO_ENDPOINT_API_KEY (the custom-endpoint fallback)"


def _pool_has(path: Path, name: str) -> bool:
    """Whether `path` already carries an entry called `name` (False when there is no pool yet)."""
    from wmo.providers.pool import load_pool

    if not path.is_file():
        return False
    try:
        return any(entry.name == name for entry in load_pool(path).models)
    except (ValueError, FileNotFoundError):
        # An unreadable pool is upsert_pool_entry's error to raise, with its own message; do not
        # pre-empt it here with a confirmation prompt about an entry we cannot see.
        return False


def _pool_disabled(path: Path, name: str) -> bool:
    """Whether `path` carries an entry called `name` with `enabled = false` (else False)."""
    from wmo.providers.pool import load_pool

    if not path.is_file():
        return False
    try:
        return any(entry.name == name and not entry.enabled for entry in load_pool(path).models)
    except (ValueError, FileNotFoundError):
        return False


def _confirm_replace(path: Path, name: str) -> bool:
    """Confirm repointing an existing pool handle; a non-interactive run declines."""
    try:
        return Confirm.ask(f"Replace the existing '{name}' entry in {path}?", default=False)
    except EOFError:
        return False


# ------------------------------------------------------------------ artifact loading boundary
# `fit` and `report` are handed paths a user typed, and the loaders behind them raise pathlib and
# pydantic errors. Both artifacts come through here so a wrong path, a truncated file, or the two
# `report` positionals in the wrong order is a usage error naming the fix, instead of the
# traceback every other input in this file is already careful not to produce.


def _reads_as(model: type[OutcomeMatrix] | type[RoutingPolicy], path: Path) -> bool:
    """Whether `path` parses as the OTHER artifact, which is what a swapped pair looks like."""
    try:
        model.model_validate_json(path.read_bytes())
    except (ValidationError, OSError):
        return False
    return True


def _load_matrix(matrix_file: str) -> tuple[OutcomeMatrix, str]:
    """The outcome matrix at `matrix_file`, with the digest provenance a fit stamps."""
    from wmo.optimize.outcomes import load_matrix_with_digest

    path = Path(matrix_file)
    try:
        return load_matrix_with_digest(path)
    except FileNotFoundError as exc:
        raise typer.BadParameter(
            f"no outcome matrix at {path}; `wmo optimize route sweep <world-model>` measures the "
            f"pool and writes one (its --out, default {DEFAULT_MATRIX_FILENAME})"
        ) from exc
    except OSError as exc:
        raise typer.BadParameter(f"cannot read the outcome matrix at {path}: {exc}") from exc
    except ValidationError as exc:
        from wmo.optimize.policy import RoutingPolicy

        if _reads_as(RoutingPolicy, path):
            raise typer.BadParameter(
                f"{path} holds a fitted policy, not an outcome matrix. The matrix is what "
                "`wmo optimize route sweep` writes, and it comes FIRST: "
                "`wmo optimize route report <matrix.json> <policy.json>`"
            ) from exc
        raise typer.BadParameter(f"{path} is not a readable OutcomeMatrix: {exc}") from exc


def _load_policy(policy_file: str) -> RoutingPolicy:
    """The fitted policy at `policy_file`."""
    from wmo.optimize.policy import RoutingPolicy

    path = Path(policy_file)
    try:
        return RoutingPolicy.load(path)
    except FileNotFoundError as exc:
        raise typer.BadParameter(
            f"no policy file at {path}; fit one with "
            "`wmo optimize route fit <matrix.json> --kind knn`"
        ) from exc
    except (OSError, UnicodeDecodeError) as exc:
        # `RoutingPolicy.load` decodes before pydantic sees anything, so bytes that are not UTF-8
        # (a truncated download, a `.npz` bank passed as the policy) raise UnicodeDecodeError,
        # which is a ValueError and NOT a ValidationError. `_load_matrix` needs no such clause:
        # `load_matrix_with_digest` hands raw bytes to pydantic, which reports undecodable input
        # as a ValidationError.
        raise typer.BadParameter(f"cannot read the policy at {path}: {exc}") from exc
    except ValidationError as exc:
        from wmo.optimize.outcomes import OutcomeMatrix

        if _reads_as(OutcomeMatrix, path):
            raise typer.BadParameter(
                f"{path} holds an outcome matrix, not a fitted policy. The policy is what "
                "`wmo optimize route fit` writes, and it comes SECOND: "
                "`wmo optimize route report <matrix.json> <policy.json>`"
            ) from exc
        raise typer.BadParameter(f"{path} is not a readable routing policy: {exc}") from exc


@route_app.command("fit")
def fit(
    matrix_file: str = typer.Argument(..., help="OutcomeMatrix JSON (closed-loop eval output)."),
    kind: str = typer.Option(
        "knn",
        "--kind",
        help="knn (guarded nearest-neighbor evidence, the validated champion, and what `tune`, "
        "`sweep`'s handoff and the docs all assume) | rank (Avengers cluster ranks).",
    ),
    out: str = typer.Option(
        _POLICY_FILENAME, "--out", help="Where to write the fitted policy JSON."
    ),
    fallback: str = typer.Option(
        None,
        "--fallback",
        help="(knn) Baseline model every request uses unless the evidence says otherwise. "
        "Default: the best single model on the fit set.",
    ),
    z: float = typer.Option(
        0.5,
        "--z",
        min=0.0,
        help="(knn) Confidence knob: standard errors of paired evidence a pick must clear to "
        "leave the fallback (doubled when it is also pricier). Higher = stricter = more "
        "requests stay on the fallback; 0 routes on any positive difference.",
    ),
    rag_num: int = typer.Option(50, "--rag-num", min=1, help="(knn) Neighbor budget."),
    rag_thres: float = typer.Option(
        0.95,
        "--rag-thres",
        min=0.0,
        max=1.0,
        help="(knn) Keep neighbors above this fraction of the rag-num-th best similarity.",
    ),
    min_pairs: int = typer.Option(
        8, "--min-pairs", min=0, help="(knn) Neighbors scored on both sides before routing away."
    ),
    floor_q: float = typer.Option(
        0.05,
        "--floor-q",
        min=0.0,
        max=1.0,
        help="Novelty floor quantile: abstain to the fallback when a query's best bank "
        "similarity is below this quantile of the bank's own nearest-neighbor sims "
        "(coverage/robustness knob for task drift; 0 = off, the exact validated champion).",
    ),
    se_floor: bool = typer.Option(
        True,
        "--se-floor/--no-se-floor",
        help="(knn) Floor the guard's standard error on thin neighborhoods (small-bank safety).",
    ),
    clusters: int = typer.Option(64, "--clusters", min=1, help="k-means cluster count."),
    seed: int = typer.Option(42, "--seed", help="Clustering seed."),
    top_k_clusters: int = typer.Option(2, "--top-k-clusters", min=1),
    beta: float = typer.Option(6.0, "--beta", help="Cluster softmax sharpness."),
    cost_weight: float = typer.Option(
        0.0,
        "--cost-weight",
        min=0.0,
        help="Quality/cost knob: reward points paid per average-call-cost unit (0 = pure "
        "accuracy ranking, the Avengers reference behavior).",
    ),
    embedder: str = typer.Option(
        "auto",
        "--embedder",
        help="auto | hashing | openai | azure. auto uses the Azure text-embedding-3-large "
        f"{' and '.join(_AZURE_EMBEDDER_ENV)} are set, and hashing otherwise; either way it says "
        "which one it picked. OpenAI and Azure CALL PAID EMBEDDING APIS billed to the selected "
        "resource; --embedder hashing keeps the fit offline and free.",
    ),
    dim: int = typer.Option(
        None,
        "--dim",
        min=1,
        help="Embedding dimension, sent as the request's `dimensions`. Default: the resolved "
        f"model's native width ({_HASHING_EMBEDDER_DIM} hashing, {_AZURE_EMBEDDER_DIM} "
        "text-embedding-3-large). Set it only to reduce a model's output deliberately.",
    ),
    deployment: str = typer.Option(
        None,
        "--deployment",
        help="Embedding model for openai, or deployment name for azure.",
    ),
    endpoint: str = typer.Option(None, "--endpoint", help="(azure) resource endpoint."),
    api_key_env: str = typer.Option(
        None, "--api-key-env", help="(openai or azure) env var holding the account key."
    ),
    compressor: str = typer.Option(
        None,
        "--compressor",
        help="D-COMPRESS: compressor id the endpoint applies before routing "
        f"({COMPRESSOR_IDS_HELP}). Default: compression off.",
    ),
    aggressiveness: float = typer.Option(
        0.0,
        "--aggressiveness",
        min=0.0,
        max=1.0,
        help="Compressor-defined dial in [0, 1]: 0.0 is a no-op and higher never removes "
        "less, but it is not an exact removal fraction (the achieved ratio is reported per "
        "call). Only meaningful with --compressor.",
    ),
) -> None:
    """Fit a routing policy on an outcome matrix (kNN evidence or Avengers cluster ranks).

    `--kind knn` is the product router and what `wmo optimize model` fits. `--kind rank` is a
    retained research direction (a faithful Avengers replication kept for comparison); the
    staged pipeline never fits it and no served endpoint carries one, so choose it only to
    measure against the champion.
    """
    from wmo.optimize.compression import (
        CompressingEmbedder,
        compression_signature,
        resolve_compression,
        same_compression,
    )
    from wmo.optimize.knn import fit_knn_artifact
    from wmo.optimize.outcomes import ROUTER_SPLIT_VERSION, split_router_scenarios
    from wmo.optimize.policy import embedder_provenance, probe_embedder, resolve_embedder
    from wmo.optimize.routing import evaluate_policy, fit_rank_policy, rerank_policy

    if kind not in ("rank", "knn"):
        raise typer.BadParameter(f"unknown kind '{kind}'; use knn or rank")
    matrix, source = _load_matrix(matrix_file)
    if not any(outcome.scored for outcome in matrix.outcomes):
        # `sweep` already exits 1 saying "fitting will fail" on this matrix, so it is a state a
        # user arrives here from: answer it the way sweep does instead of letting the fitter's
        # own ValueError out (the rank one names no remedy at all).
        raise typer.BadParameter(
            f"no cell in {matrix_file} carries a reward, so there is nothing to fit: read the "
            "`error` field of a row to see what broke, fix it, then re-run "
            "`wmo optimize route sweep <world-model>`"
        )
    try:
        spec, resolution = resolve_embedder(
            embedder, dim=dim, deployment=deployment, endpoint=endpoint, api_key_env=api_key_env
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    # Printed before the fit, not after: the embedder decides what the policy can route on, and an
    # operator who meant to fit on semantic vectors should see that it fell back BEFORE paying for
    # the fit and reading an accuracy number that quietly came from hashed features.
    _console.print(resolution)
    out_path = Path(out)
    if rag_thres <= 0.0:
        # typer's min is inclusive but the artifact field requires > 0; fail before the fit
        # writes a sidecar it will then abandon.
        raise typer.BadParameter("--rag-thres must be greater than 0")
    # One throwaway embedding BEFORE the bulk work: an unreachable or embedding-less resource is
    # a usage error at the boundary here, instead of a traceback from inside the fit.
    try:
        probe_embedder(spec)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if compressor is None and aggressiveness > 0.0:
        raise typer.BadParameter("--aggressiveness needs --compressor to apply it")
    compression = None
    if compressor is not None:
        try:
            # Fail before the fit spends anything: model_copy below skips validators, and an
            # unservable compressor would otherwise only surface when serving mounts the result.
            compression = resolve_compression(compressor, aggressiveness)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    # The rewards in this matrix were produced under SOME compression config, and a joint fit is
    # only joint if that config is the one being fitted. `--compressor` moves the fit-side
    # representation (embeddings), but it cannot retroactively change what the episodes ran
    # under: fitting a compressed policy over uncompressed rewards would stamp an arm that was
    # never measured. Checked both directions, since compressed rewards under a raw fit is the
    # same mistake mirrored.
    measured = matrix.measured_compression()
    if not same_compression(measured, compression):
        raise typer.BadParameter(
            f"this matrix's rewards were measured with {compression_signature(measured)}, but "
            f"the fit would stamp {compression_signature(compression)}. Rewards cannot be "
            "recompressed after the fact, so measure the arm you intend to serve: "
            "`wmo optimize route sweep <model> --compressor <id> --aggressiveness <a>` writes a "
            "matrix whose episodes actually ran that way (one matrix per arm)."
        )
    try:
        router_split = split_router_scenarios(matrix.scenario_ids())
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    fit_ids = list(router_split.fit_ids)
    if kind == "knn":
        if cost_weight > 0.0:
            raise typer.BadParameter(
                "--cost-weight re-ranks cluster evidence and applies to --kind rank only; a knn "
                "policy trades cost through its dial instead: fit it, then "
                "`wmo optimize route tune <policy.json> --cost-quality <0..1>`"
            )
        try:
            fitted = fit_knn_artifact(
                matrix,
                out_path=out_path,
                matrix_source=source,
                embedder=spec,
                fit_ids=fit_ids,
                fallback=fallback,
                z=z,
                rag_num=rag_num,
                rag_thres=rag_thres,
                min_pairs=min_pairs,
                se_floor=se_floor,
                floor_q=floor_q,
                compression=compression,
            )
        except ValueError as exc:
            # What the fitter can still refuse once the matrix loads: an unknown --fallback, or a
            # fit split whose own rows are all unscored. Both are about the arguments.
            raise typer.BadParameter(str(exc)) from exc
        print_knn_fit(_console, fitted, out=out, z=z)
        return
    built = spec.build()  # ONE embedder for fit and evaluation; azure would otherwise embed twice
    if compression is not None:
        # Representation consistency: the cluster centroids have to live in the geometry of the
        # text serving will embed, which is the COMPRESSED text (see `fit_knn_artifact`, which
        # applies the same rule to the bank on the knn path).
        built = CompressingEmbedder(built, compression)
    try:
        policy = fit_rank_policy(
            matrix,
            fit_ids=fit_ids,
            embedder=spec,
            n_clusters=clusters,
            seed=seed,
            top_k_clusters=top_k_clusters,
            beta=beta,
            fitted_from=(
                f"{source} split={ROUTER_SPLIT_VERSION} "
                f"rank seed={seed} k={clusters} topk={top_k_clusters} beta={beta:g} "
                f"cost_weight={cost_weight:g} {embedder_provenance(spec)}"
            ),
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if cost_weight > 0.0:
        policy = rerank_policy(policy, cost_weight=cost_weight)
    if compression is not None:
        # Stamped as BOTH halves of the contract: what this endpoint serves, and what its
        # evidence was fitted under. They are the same config here by construction (the fit just
        # embedded through it), which is exactly what the mount gate re-checks. The knn path
        # stamps inside `fit_knn_artifact`, which saves and returns before this line.
        policy = policy.model_copy(
            update={"compression": compression, "fit_compression": compression}
        )
    policy.save(out_path)
    result = evaluate_policy(policy, matrix, fit_ids, embedder=built)
    _console.print(
        f"[green]✓[/green] fitted {len(policy.clusters)} clusters over "
        f"{result.scenarios} scenarios -> {out}\n"
        f"  fit-set accuracy {result.accuracy:.4f}, cost/scenario ${result.cost_per_scenario:.5f}"
    )


def print_knn_fit(console: Console, fitted: KnnFitOutcome, *, out: str, z: float) -> None:
    """Report a written knn policy: where its evidence is, and what it scored in-sample."""
    console.print(
        f"[green]✓[/green] fitted knn policy over {fitted.scenarios} scenarios -> {out}\n"
        f"  bank {fitted.bank_path}, fallback {fitted.policy.default_model}, z={z}\n"
        f"  routed away from the fallback {fitted.routed_share:.1%} of the time; cost/scenario "
        f"${fitted.cost_per_scenario:.5f}\n"
        f"  fit-set accuracy {fitted.fit_accuracy:.4f} is IN-SAMPLE (every request retrieves its "
        "own row); measure on held-out scenarios with `wmo optimize route report`"
    )


@route_app.command("pin")
def pin(
    world_model: str = typer.Argument(
        None, help="Built world model whose endpoint serves this policy. Default: the only one."
    ),
    model: str = typer.Option(
        ...,
        "--model",
        help="Pool entry every request goes to (a `wmo optimize route student` name).",
    ),
    pool: str = typer.Option(
        _DEFAULT_POOL_PATH, "--pool", help="Candidate pool TOML to snapshot into the policy."
    ),
    root: str = typer.Option(ARTIFACT_DIR, "--root", help="Project dir holding the built models."),
    out: str = typer.Option(
        None, "--out", help="Override where the policy JSON lands (default: the model's own dir)."
    ),
    yes: bool = typer.Option(
        False, "--yes", help="Skip the confirmation when a policy is already installed."
    ),
) -> None:
    """Serve one pool model as an endpoint, with no matrix and no fit.

    A `kind="static"` policy sends every request to `--model`, which is all a single distilled
    student needs to be reachable through the OpenAI-compatible endpoint:

        wmo optimize route student .wmo/distill/support --input-per-mtok 0.1 --output-per-mtok 0.4
        wmo optimize route pin support --model student
        wmo serve --name support

    The policy is written to the world model's artifact dir, because that is where `wmo serve`
    looks for one. This is the honest "before" state the routing story is told against: a static
    endpoint has learned nothing and saves nothing, and `GET /v1/endpoints/<name>/savings` will
    say so. Replace it with `wmo optimize route fit` on a real outcome matrix to let the router
    choose per request.
    """
    from wmo.optimize.policy import RoutingPolicy
    from wmo.providers.pool import load_pool

    store = WorldModelStore(root)
    try:
        model_dir = store.resolve(world_model)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except ValueError as exc:
        # `resolve` says "pass --name", the option `wmo serve`/`play`/`demo` carry. This command
        # has no --name (its --model is the POOL entry), so say what a user of it actually types.
        names = store.list_names()
        raise typer.BadParameter(
            f"multiple world models built ({', '.join(names)}); name one as the WORLD_MODEL "
            f"argument, e.g. `wmo optimize route pin {names[0]} --model {model}`"
        ) from exc
    pool_path = Path(pool)
    try:
        roster = load_pool(pool_path)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except ValueError as exc:
        raise typer.BadParameter(f"cannot read the pool at {pool_path}: {exc}") from exc
    active = roster.enabled_models()
    if all(entry.name != model for entry in active):
        if any(entry.name == model for entry in roster.models):
            raise typer.BadParameter(
                f"pool model '{model}' is disabled (enabled = false) in {pool_path}; flip it "
                "back on to pin it"
            )
        available = ", ".join(entry.name for entry in active)
        raise typer.BadParameter(
            f"no pool model named '{model}' in {pool_path}; available: {available}"
        )
    out_path = Path(out) if out else model_dir / _POLICY_FILENAME
    if out and out_path.resolve() != (model_dir / _POLICY_FILENAME).resolve():
        # The foot-gun that bit both bench-defaults lanes (2026-07-29): an --out
        # anywhere but <model dir>/policy.json succeeds, prints the same cheerful
        # line, and leaves the file serving actually reads holding whatever policy
        # it held before (a different FILENAME in the right dir misses identically).
        # The pin still lands where asked; the operator is told serving will not
        # see it.
        _console.print(
            f"[yellow]![/yellow] --out is outside {model_dir}; `wmo serve --name "
            f"{model_dir.name}` and GET /config read {model_dir / _POLICY_FILENAME}, "
            "which this pin does NOT update"
        )
    if out_path.is_file() and not yes and not _confirm_overwrite(out_path):
        _console.print(f"left {out_path} in place")
        raise typer.Exit(0)
    policy = RoutingPolicy(
        kind="static",
        default_model=model,
        # Only the enabled roster travels: the policy's pool is what serving may construct
        # providers for, and a turned-off candidate must not become reachable through an
        # endpoint pinned after the operator turned it off.
        pool=active,
        fitted_from=f"pinned to {model} from {pool_path} (no outcome matrix)",
    )
    policy.save(out_path)
    _console.print(
        f"[green]✓[/green] pinned endpoint [bold]{model_dir.name}[/bold] to "
        f"[bold]{model}[/bold] -> {out_path}\n"
        f"  every request goes to {model}; nothing is measured and nothing is saved yet\n"
        f"  serve it: wmo serve --name {model_dir.name}\n"
        "  to let the router choose per request instead, replace this with "
        "`wmo optimize route fit <matrix.json>`",
        soft_wrap=True,
    )


def _confirm_overwrite(path: Path) -> bool:
    """Confirm replacing an installed policy; a non-interactive run declines.

    Worth asking about: the file being replaced may be a fitted knn policy, whose evidence bank
    sidecar this static policy will not use and does not remove.
    """
    try:
        return Confirm.ask(f"Replace the policy already at {path}?", default=False)
    except EOFError:
        return False


@route_app.command("tune")
def tune(
    policy_file: str = typer.Argument(_POLICY_FILENAME, help="Fitted knn policy JSON to re-tune."),
    cost_quality: float = typer.Option(
        ...,
        "--cost-quality",
        min=0.0,
        max=1.0,
        help="The endpoint's one dial: 0.0 = max quality, 1.0 = max savings. 0.25 is the "
        "shipped default. See the anchor table this command prints for what each end measured.",
    ),
) -> None:
    """Set a fitted policy's cost/quality dial in place, without refitting anything.

    The dial maps to the policy's knobs along the measured frontier (see
    `wmo.optimize.knn.apply_cost_quality`). The first successful run copies the un-tuned artifact
    to `policy.base.json` and every later run re-reads THAT, so the dial is always applied to the
    policy as fitted and sliding twice never compounds:

        wmo optimize route tune models/support/policy.json --cost-quality 0.6

    That snapshot is only a valid baseline for the fit it came from, so this command refuses to
    run when the two disagree (refit the policy and the stale snapshot must be deleted, not
    silently dialed back over the new fit). A tune that is rejected writes nothing at all, and
    every write it does make is atomic.

    The evidence bank is untouched, so this is instant. A served endpoint can be dialed without
    touching files at all: `PUT /v1/endpoints/{name}/config`.
    """
    from wmo.optimize.knn import tune_policy_dial

    try:
        dialed = tune_policy_dial(Path(policy_file), cost_quality)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    print_dial(_console, dialed)


def print_dial(console: Console, dialed: DialResult) -> None:
    """Report an applied dial position against the frontier that was actually measured."""
    from wmo.optimize.knn import COST_QUALITY_ANCHORS

    knobs = dialed.knobs
    console.print(
        f"[green]✓[/green] cost_quality={dialed.cost_quality:g} "
        f"({dialed.named_point}) -> {dialed.policy_path}\n"
        f"  knobs: floor_q={knobs.floor_q:g}, cost knob lam={knobs.pick_lam:g}, "
        f"guard={knobs.guard_mode}, z={knobs.knn_z:g}\n"
        f"  as fitted: {dialed.base_path}\n"
        f"  measured on routerbench-ours9 (5 held-out splits, vs the best single model):"
    )
    for anchor in COST_QUALITY_ANCHORS:
        marker = "->" if anchor.cost_quality == dialed.cost_quality else "  "
        console.print(
            f"  {marker} {anchor.cost_quality:<5g} {anchor.quality_delta_points:+.2f}pt "
            f"@ {anchor.cost_delta_percent:+.1f}% cost"
            + (f"  [dim]{anchor.named_point}[/dim]" if anchor.named_point != "Custom" else "")
        )


@route_app.command("report")
def report(
    matrix_file: str = typer.Argument(..., help="OutcomeMatrix JSON with held-out scenarios."),
    policy_file: str = typer.Argument(..., help="Fitted policy JSON."),
    baseline: str = typer.Option(
        ...,
        "--baseline",
        # The doubled brackets are escaped for the same reason `sweep --pool`'s are: typer
        # renders help through rich markup, which otherwise prints an empty pair.
        help="Pool entry HANDLE to compare against: the `name` of a \\[\\[model]] table in the "
        "matrix's pool, NOT the model id. Normally the frontier candidate.",
    ),
    endpoint: str = typer.Option("endpoint", "--endpoint", help="Endpoint id for the report."),
    out: str = typer.Option("report.json", "--out", help="Where to write the report JSON."),
    provenance: str = typer.Option(
        _WM_SIMULATED,
        "--provenance",
        help=f"How this matrix's rewards were produced: {_WM_SIMULATED} (closed-loop against a "
        f"world model, the default) or {_REAL_EPISODE} (episodes of the real benchmark). It rides "
        "on the pareto curve and must never be wrong: consumers refuse to blend the two.",
    ),
    judge: str = typer.Option(
        _DEFAULT_WM_JUDGE,
        "--judge",
        help="What scored the episodes, printed beside every rendering of the curve. Pass the "
        "real scorer for a real-benchmark matrix (for example \\[tau2 reward]).",
    ),
    scenario_label: str = typer.Option(
        "",
        "--scenario-label",
        help="The report's customer-facing sentence describing WHAT was measured. Defaults to the "
        "world-model phrasing ('reconstructed from your traces'), which is false for a real "
        "benchmark, so pass the truth there (for example 'on the 20 pinned tau2-bench eval "
        "tasks').",
    ),
) -> None:
    """Build the improvement report for a fitted policy over a matrix."""
    from wmo.optimize.pareto import (
        PARETO_FILENAME,
        REAL_EPISODE,
        WM_SIMULATED,
        held_out_curve,
    )
    from wmo.optimize.policy import write_artifact_atomically
    from wmo.optimize.report import build_report

    if provenance not in {WM_SIMULATED, REAL_EPISODE}:
        # A typo here would silently label real measurements as simulated, which is the one
        # mistake the curve's provenance field exists to prevent.
        raise typer.BadParameter(
            f"--provenance must be {WM_SIMULATED} or {REAL_EPISODE}, not {provenance!r}"
        )
    matrix, matrix_source = _load_matrix(matrix_file)
    policy = _load_policy(policy_file)
    try:
        improvement = build_report(
            matrix,
            policy,
            baseline=baseline,
            endpoint=endpoint,
            generated_at=datetime.now(tz=UTC).isoformat(),
            scenario_label=scenario_label or None,
        )
    except KeyError as exc:
        # `--baseline` is a pool entry handle; the KeyError already lists the ones this matrix
        # has. `str()` on a KeyError quotes its own argument, so unwrap it.
        raise typer.BadParameter(
            f"{exc.args[0]}. --baseline names a pool entry handle (the `name` of a [[model]] "
            "table), not a model id."
        ) from exc
    except FileNotFoundError as exc:
        # A knn policy carries its evidence in a sidecar; `knn_bank` says how to restore it.
        raise typer.BadParameter(str(exc)) from exc
    except ValueError as exc:
        # Nothing scored on both sides, so there is no paired comparison to report.
        raise typer.BadParameter(str(exc)) from exc
    # mkdir + atomic, exactly as `fit --out` and `pin --out` write their policies: a report whose
    # parent directory does not exist must not throw away the work that produced it.
    write_artifact_atomically(Path(out), improvement.model_dump_json(indent=2).encode("utf-8"))
    # The measured cost/quality curve rides beside every report (D-PARETO): GET /config
    # serves it from the model dir so the platform's graph renders this workload's frontier.
    try:
        curve = held_out_curve(matrix, policy, judge=judge, provenance=provenance)
        pareto_out = Path(out).parent / PARETO_FILENAME
        write_artifact_atomically(pareto_out, curve.model_dump_json(indent=2).encode("utf-8"))
        _console.print(f"[green]✓[/green] pareto curve -> {pareto_out}")
        # Same foot-gun as `pin --out`: serving loads policy.json and pareto.json from ONE
        # model dir, so a curve written apart from the policy it describes is a curve
        # `wmo serve` and GET /config never show. Succeeding silently here is what hid it.
        if Path(policy_file).resolve().parent != pareto_out.resolve().parent:
            _console.print(
                f"[yellow]![/yellow] the curve landed apart from {policy_file}; serving reads "
                "pareto.json from the same directory as the policy it mounts, so point --out "
                "there for the endpoint to show this curve"
            )
    except (ValueError, FileNotFoundError) as exc:
        _console.print(f"[yellow]![/yellow] pareto curve skipped: {exc}")
    headline = improvement.headline
    _console.print(
        f"[green]✓[/green] report -> {out}\n"
        f"  routed acc {headline.accuracy:.4f} @ ${headline.cost_per_run_usd:.5f}/run vs "
        f"{baseline} {headline.baseline_accuracy:.4f} @ ${headline.baseline_cost_per_run_usd:.5f}"
    )
    in_sample = _in_sample_warning(policy, matrix_source)
    if in_sample is not None:
        _console.print(in_sample)


def _in_sample_warning(policy: RoutingPolicy, matrix_source: str) -> str | None:
    """The caveat for a report measured on the very matrix the policy was fitted on.

    Since the router split (#308), `build_report` excludes the policy's recorded fit scenarios,
    so a report over the fit matrix IS held out whenever that split is recoverable - the note
    says so, with the count. The in-sample WARNING remains only for a policy that records no
    split and whose evidence cannot name one: those numbers retrieve their own rows. The digest
    in `fitted_from` is an identity rather than a label (`load_matrix_with_digest`), so the
    collision is detectable even when the matrix was renamed or moved after the fit, and a
    matrix with the same path but different bytes does not trip it.
    """
    # `load_matrix_with_digest` appends the marker LAST, so split from the right: a matrix under a
    # content-addressed directory (`artifacts/sha256=.../matrix.json`) carries the marker in its
    # path too, and splitting from the left would read that one and silently drop the warning.
    _, mark, digest = matrix_source.rpartition(_MATRIX_DIGEST_MARK)
    stamped = policy.fitted_from or ""
    if not mark or not digest or f"{_MATRIX_DIGEST_MARK}{digest}" not in stamped:
        return None
    fit_ids = set(policy.fit_scenario_ids)
    if not fit_ids and policy.kind == "knn":
        # The same recovery `build_report` uses for legacy kNN artifacts: their evidence bank
        # names the fit scenarios even when the policy predates recording them.
        try:
            fit_ids = set(policy.knn_bank().scenario_ids)
        except (FileNotFoundError, ValueError):
            fit_ids = set()
    if fit_ids:
        # Same matrix as the fit, but the report excluded the fit scenarios (the split is
        # recorded on the policy), so the numbers above ARE held out. Say what happened instead
        # of contradicting the report's own label.
        return (
            f"note: same matrix as the fit ({_MATRIX_DIGEST_MARK}{digest}); the "
            f"{len(fit_ids)} fit scenario(s) were excluded, so the numbers above are over "
            "held-out scenarios only."
        )
    return (
        f"[yellow]warning[/yellow] this policy was FITTED on this matrix "
        f"({_MATRIX_DIGEST_MARK}{digest}) and records no fit split, so these numbers are "
        "IN-SAMPLE, not held out: every request retrieves its own row. Sweep a second matrix "
        "over scenarios the fit never saw and report against that one."
    )


@route_app.command("push")
def push(
    policy_file: str = typer.Argument(_POLICY_FILENAME, help="Fitted policy JSON to install."),
    endpoint: str = typer.Option(
        ...,
        "--endpoint",
        help="Hosted endpoint slug to install onto (the `model` a customer's client sends).",
    ),
    org: str | None = typer.Option(
        None,
        "--org",
        help="Organization id (default: the login's, or $WMO_PLATFORM_ORG).",
    ),
    report_file: str | None = typer.Option(
        None,
        "--report",
        help="Improvement report JSON to publish with the policy (see `route report`).",
    ),
) -> None:
    """Install a fitted policy on a hosted endpoint, so serving actually uses it.

    The last link in the chain. `fit` writes a policy that only this machine can see;
    an endpoint created on the platform serves a `static` policy until something
    replaces it. This is that something:

        wmo optimize route push models/support/policy.json --endpoint support-prod

    A knn policy is TWO artifacts, and this sends both: the JSON plus the `.npz`
    evidence bank beside it, resolved from the policy's own `knn_bank_path` rather
    than guessed, so a renamed sidecar is a local error instead of a server refusal.
    Sending the policy alone would store a row that validates and cannot serve.

    The endpoint keeps its id, name, and URL, so a customer's client is unaffected by
    the swap, and live pods pick the new policy up on their own.
    """
    from wmo.optimize.policy import RoutingPolicy

    policy_path = Path(policy_file)
    if not policy_path.is_file():
        raise typer.BadParameter(
            f"no policy at {policy_path} (`wmo optimize route fit` writes one)"
        )
    try:
        policy = RoutingPolicy.load(policy_path)
    except (OSError, ValidationError, ValueError) as exc:
        raise typer.BadParameter(f"{policy_path} is not a routing policy: {exc}") from exc

    bank_path: Path | None = None
    if policy.kind == "knn":
        bank_path = policy.bank_path()
        if not bank_path.is_file():
            # Checked here, not left to the server: the policy names its own sidecar,
            # so a missing one means the local artifact pair is broken and pushing
            # would only turn that into a 400 after uploading nothing useful.
            raise typer.BadParameter(
                f"{policy_path} is a knn policy whose evidence bank is missing at "
                f"{bank_path}; a knn policy is served together with its sidecar, so "
                "copy it beside the policy or refit with `wmo optimize route fit --kind knn`"
            )
    report_path = Path(report_file) if report_file is not None else None
    if report_path is not None:
        if not report_path.is_file():
            raise typer.BadParameter(
                f"no report at {report_path} (`wmo optimize route report` writes one)"
            )
        try:
            json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            # Same rule as the sidecar check: a broken local artifact fails here,
            # not as a server refusal after the upload already happened.
            raise typer.BadParameter(f"{report_path} is not readable JSON: {exc}") from exc

    # Imported here, not at module scope: `platform_cmds` builds its own command
    # surface at import time, and pulling that in for one command changed behavior
    # in unrelated `route` commands (15 of this module's tests went red).
    from wmo.cli.platform_cmds import _connected, _require_connection

    # Sized before the install: a stat after a SUCCESSFUL install would make the
    # whole command read as failed if anything removed the local file meanwhile.
    size = f", bank {bank_path.stat().st_size / 1024:.0f}KiB" if bank_path is not None else ""

    credentials, org_id = _require_connection(org)
    with _connected(credentials, "Could not install the policy") as client:
        client.install_endpoint_policy(org_id, endpoint, policy_path, bank_path, report_path)

    _console.print(
        f"[green]✓[/green] installed {policy.kind} policy on [bold]{endpoint}[/bold]{size}\n"
        f"  from: {policy_path}\n"
        f"  serving picks it up without a restart; `wmo runs` and the endpoint's "
        f"telemetry show what it routes."
    )
