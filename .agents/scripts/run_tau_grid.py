"""Chunked, resumable runner for the canonical tau grid (DECISIONS.md 2026-07-27, option A).

Buys the joint-policy evidence base task 6 needs: 3 arms (identity / truncate at the endpoint's
matched ACHIEVED ratio / llmlingua2-endpoint) x the candidate pool x the world model's test-band
scenarios x 1 episode, at max-steps 20. It exists instead of three `wmo optimize route sweep`
calls because the sweep path has no concurrency and persists ONLY at sweep end, so a single
transport fault twelve hours in loses every cell it had already paid for.

MEASURED SIZE, which is smaller than option A's forecast assumed: `scenarios_from_traces`
collapses a task repeated across traces into ONE scenario, and this corpus's 1033 traces carry
only 126 distinct task prompts (102 train / 14 val / 20 test). So `--scenarios 45` against the
test band yields 20, and the grid is 20 scenarios x the pool x 3 arms rather than 1215 cells.
Raising `--episodes` is the lever that recovers episode-weighted power without leaving the
leak-free band; nothing here silently pads the cut to reach a target count.

Everything measured here goes through the PUBLIC library (`wmo.optimize.sweep`), never a private
copy of it. The only thing this script owns is durability:

- CHUNKING. Each arm is `ceil(cut / --chunk-size)` chunks, each chunk one `execute_sweep` over the
  full pool, its `OutcomeMatrix` saved the moment it finishes. The scenario cut is the sweep's own
  deterministic cut (`plan_sweep`), sliced; nothing here decides which scenarios are measured.
- RESUME. A chunk whose matrix file loads clean is skipped, so re-running continues where the
  last process stopped, and three staggered `--arm` processes can share one grid directory.
- CONCURRENCY. `--chunks A:B` narrows a process to a half-open chunk-index range, so ONE arm can
  also be driven by several processes at once. Chunk matrices, per-chunk retry records, and
  world-model usage records are all keyed per chunk index, so disjoint ranges never touch the same
  file and no lock is needed. The ledger is append-only (one `open(..., "a")` and one
  newline-terminated JSON line per write, no read-modify-write), and the spend cap is evaluated by
  RE-READING it before each chunk so every process sees its siblings' bills. The documented cost of
  going lock-free: N concurrent processes can each pass the cap check and then each buy a chunk, so
  the grid may overshoot the cap by up to one chunk per running process.
- RETRY. After an arm's chunks, cells left unscored by a TRANSIENT fault (the Bedrock
  InternalServerExceptions the calibration probe saw, throttles, 5xx, timeouts) are re-executed
  once, individually. A cell that fails twice keeps its error row: an unscored cell is evidence
  about the infrastructure, and defaulting it to a reward would read a 500 as incapability.
- COHORT. Every ledger line and merged matrix carries the main tip sha, max-steps, and episodes
  the cells were bought under. A chunk cannot be appended to a grid stamped with a different tip:
  a new cohort gets a new directory, because rows measured under two harnesses are not one matrix.
  The candidate pool is pinned the same way, as a copy inside the grid directory, because
  `.wmo/pool.toml` is live local state other lanes edit (it gained two candidates during this
  runner's own smoke).
- SPEND. Both halves of each chunk's bill (candidate side, world-model serve/judge side) land in
  the ledger, and the run stops cleanly between chunks at the $850 cap Silen set. Reaching the cap
  stops the run and reports; it never trims an arm to fit.

NO HANG DETECTION, deliberately. Nothing here imposes a per-call, per-episode or per-chunk
deadline: the only clock is a `time.monotonic()` bracket around each `execute_sweep` whose result
goes into the ledger's `wall_s` and nowhere else. So a serverless candidate's cold start (kimi-k3
was measured at 51 seconds on its first call) is recorded as wall time, never mistaken for a stall
and never killed. If a backend's own SDK gives up and reports a timeout, that surfaces as the
cell's error and the retry pass gives it one more attempt, by which point the model is warm.

TRUNCATE CALIBRATION. `truncate` is the control the learned compressor has to beat, so it is
matched on the endpoint's ACHIEVED keep ratio, not on a nominal dial (`CompressionConfig`: the
achieved ratio of a fixed-threshold compressor is an outcome). This measures the endpoint on a
sample of TRAIN-band transcripts, searches truncate's aggressiveness for the value whose achieved
ratio lands within +-0.05, and persists the result to `calibration.json` so every arm process in
the grid uses the same number. If the endpoint is unreachable the compressed arms STOP: a
substituted ratio would silently make the control and the method incomparable.

    # smoke first (<= $2): 2 cheap models, 2 scenarios, identity + endpoint arms
    uv run python .agents/scripts/run_tau_grid.py --smoke --env-file <compressor-env>

    # the buy, one process per arm, staggered (see --help for the cap and grid dir)
    uv run python .agents/scripts/run_tau_grid.py --arm identity
    uv run python .agents/scripts/run_tau_grid.py --arm truncate --env-file <compressor-env>
    uv run python .agents/scripts/run_tau_grid.py --arm llmlingua2-endpoint --env-file <env>

    # escalating one arm to two processes, only while the Bedrock error rate stays clean
    uv run python .agents/scripts/run_tau_grid.py --arm identity --chunks :2
    uv run python .agents/scripts/run_tau_grid.py --arm identity --chunks 2:

Concurrency has a ceiling that is not this script's: the world model's own serve and judge calls
are ONE Bedrock model (opus) and throttle as one bucket, so past roughly 4-6 processes the extra
parallelism converts into throttles, which land as unscored cells and turn into retry-pass work.

Credentials come from the gitignored `.env` plus the ambient environment: Azure keys and the
Tinker token from `.env`, Anthropic from `ANTHROPIC_API_KEY`, Bedrock from the AWS credential
chain, and the compressor endpoint's URL/token/pinned certificate from whichever `.env` holds them
(`--env-file`). Nothing here prints a secret.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import subprocess
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from wmo.config import load_env_file
from wmo.core.types import ActionKind, Trace
from wmo.engine import load_world_model, split_holdout
from wmo.engine.world_model import WorldModel
from wmo.env import WorldModelEnv
from wmo.env.closed_loop import scenario_id
from wmo.env.llm_agent import DEFAULT_HISTORY_CHARS
from wmo.env.scenarios import Scenario
from wmo.ingest import get_adapter
from wmo.optimize.compression import (
    CompressionConfig,
    compress_segments,
    estimate_tokens,
    get_compressor,
)
from wmo.optimize.compression_endpoint import CompressorEndpointError, register_endpoint_compressor
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.sweep import (
    CostLine,
    SweepError,
    SweepPlan,
    SweepRun,
    coverage,
    execute_sweep,
    plan_sweep,
    preflight_pool,
    resolve_config,
)
from wmo.providers.pool import ModelPool

log = logging.getLogger("tau-grid")

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------- the registered grid
# Option A's pins, as approved. They are constants rather than bare defaults so a reader can see
# the cohort the matrices claim, and every one is overridable for the smoke.
IDENTITY_ARM = "identity"
TRUNCATE_ARM = "truncate"
ENDPOINT_ARM = "llmlingua2-endpoint"
# C2 round 3: segment-SCOPED arms (compress tool observations only, never dialogue or the
# task; wmo.optimize.compression_scoped). The scoped truncate control shares the whole-context
# calibration's dial: both scoped arms compress the SAME observation spans, so matching inner
# dials matches their achieved ratios on those spans; the per-episode ACHIEVED keep is recorded
# either way and the match is verified post-hoc at the compression track's 0.05 tolerance.
SCOPED_ENDPOINT_ARM = "scoped-llmlingua2-endpoint"
SCOPED_TRUNCATE_ARM = "scoped-truncate"
ALL_ARMS = (IDENTITY_ARM, TRUNCATE_ARM, ENDPOINT_ARM, SCOPED_ENDPOINT_ARM, SCOPED_TRUNCATE_ARM)

DEFAULT_MODEL_DIR = REPO_ROOT / ".wmo" / "models" / "tau-bench"
DEFAULT_POOL = REPO_ROOT / ".wmo" / "pool.toml"
DEFAULT_TRACES = REPO_ROOT / "packages" / "environment-capture" / "tau-bench" / "traces.otel.jsonl"
DEFAULT_GRID_DIR = REPO_ROOT / ".wmo" / "jt" / "grid"
DEFAULT_SCENARIOS = 45
DEFAULT_CHUNK = 5
DEFAULT_EPISODES = 1
DEFAULT_MAX_STEPS = 20
DEFAULT_CAP_USD = 850.0

# The threshold the box serves and the docs quote (deploy/compressor-endpoint/README.md, and the
# server's own `CompressRequest.threshold` default). For this compressor the dial IS the absolute
# keep probability, so naming it here is naming the arm.
ENDPOINT_AGGRESSIVENESS = 0.5

# How close truncate's achieved keep ratio must land to the endpoint's for the control to be
# ratio-matched. The compression track's own matching tolerance.
RATIO_TOLERANCE = 0.05
CALIBRATION_SAMPLE = 20
CALIBRATION_SEED = 0
# Coarse pass then fine pass around the winner: truncate drops whole words, so its achieved ratio
# is a step function of the dial and a 0.01 grid over the whole range is both wasteful and no more
# precise than the steps themselves.
CALIBRATION_COARSE_STEP = 0.05
CALIBRATION_FINE_STEP = 0.01

# The smoke's pins: the two cheapest candidates, a short step budget, and its own grid directory
# so it can never append a row to the real cohort.
SMOKE_MODELS = ("haiku-4-5", "gpt-5.4-mini")
SMOKE_ARMS = (IDENTITY_ARM, ENDPOINT_ARM)
SMOKE_SCENARIOS = 2
SMOKE_CHUNK = 1
SMOKE_MAX_STEPS = 6
SMOKE_CAP_USD = 2.0
SMOKE_GRID_DIR = REPO_ROOT / ".wmo" / "jt" / "grid-smoke"

# A cell whose error matches one of these is worth exactly one more attempt: the fault is in the
# transport or the backend's capacity, not in the request. Everything else (a malformed pool
# entry, an auth rejection, a judge that refuses the transcript) would fail identically the
# second time and is left as measured.
_TRANSIENT_PATTERNS = (
    r"internalservererror",
    r"internalserverexception",
    r"serviceunavailable",
    r"throttl",
    r"too many requests",
    r"rate.?limit",
    r"\b429\b",
    r"\b5(?:00|02|03|04)\b",
    r"timed? ?out",
    r"timeout",
    r"connection (?:reset|error|aborted|refused)",
    # Observed in the smoke: the compressor endpoint dropped one request mid-flight. Its client
    # already retries a transport fault once, so a cell only reaches here when BOTH attempts were
    # dropped, which is still capacity rather than a bad request.
    r"server disconnected",
    r"remote end closed",
    r"eof occurred",
    r"read operation timed out",
)
_TRANSIENT = re.compile("|".join(_TRANSIENT_PATTERNS), re.IGNORECASE)

# Above this share of a chunk's cells failing transiently, the retry pass reads the failures as one
# outage rather than many blips and declines to buy the same failure once per cell.
SYSTEMIC_FAILURE_SHARE = 0.5

# The calibration probe's measured all-in cost per cell at max-steps 12 ($7.05 over 18 cells,
# DECISIONS.md 2026-07-27). Used ONLY as the cap guard's seed: once this grid has measured cells of
# its own it averages those instead, because the probe ran at a different step budget.
PROBE_USD_PER_CELL = 0.39

MERGED_MATRIX = "matrix.json"
MERGED_META = "matrix.meta.json"
LEDGER_FILE = "ledger.jsonl"
COHORT_FILE = "cohort.json"
CALIBRATION_FILE = "calibration.json"
RETRIED_FILE = "retried-{chunk}.json"


class GridStopped(RuntimeError):
    """The run stopped short of finishing, for a reason the operator has to act on."""


# --------------------------------------------------------------------------------- typed artifacts


class Cohort(BaseModel):
    """What every cell in a grid directory was bought under.

    A matrix is only evidence about the harness that produced it, so the tip sha, step budget and
    episode count are stamped once per directory and then CHECKED on every append. Mixing two
    tips in one arm would produce a matrix whose rows answer to different code, which no amount of
    later analysis can separate.
    """

    model_config = ConfigDict(extra="forbid")

    tip_sha: str
    max_steps: int
    episodes: int
    scenarios: int
    chunk_size: int
    history_chars: int
    model_dir: str
    pool_file: str
    traces_file: str
    created: str

    def mismatch(self, other: Cohort) -> str | None:
        """Why `other` may not append to this cohort, or None when the pins agree."""
        differences = [
            f"{field}: recorded {getattr(self, field)!r}, this run {getattr(other, field)!r}"
            for field in ("tip_sha", "max_steps", "episodes", "history_chars", "model_dir")
            if getattr(self, field) != getattr(other, field)
        ]
        return "; ".join(differences) if differences else None


class Calibration(BaseModel):
    """The ratio match between the learned compressor and its truncation control.

    Persisted per grid directory because three arm processes must agree on truncate's dial: two
    processes that each calibrated their own would produce two different controls and there would
    be no single ratio-matched arm to compare the method against.
    """

    model_config = ConfigDict(extra="forbid")

    sample_size: int
    sample_tokens_raw: int
    endpoint_aggressiveness: float
    endpoint_achieved_ratio: float
    searched: list[tuple[float, float]]  # (truncate aggressiveness, achieved keep ratio)
    chosen_aggressiveness: float
    chosen_achieved_ratio: float
    tolerance: float
    measured_at: str
    tip_sha: str


class LedgerLine(BaseModel):
    """One appended fact about the grid's progress and its bill.

    Append-only JSONL rather than a rewritten summary: the ledger has to survive the SIGKILL that
    the thing it is tracking might not, and a partially written line is one bad line rather than a
    lost file.
    """

    model_config = ConfigDict(extra="forbid")

    event: str  # chunk | chunk-skipped | retry | calibration | merge | stop
    arm: str
    chunk: int | None = None
    cells: int = 0
    scored: int = 0
    candidate_usd: float = 0.0
    compressor_usd: float = 0.0
    wm_usd: float = 0.0
    wall_s: float = 0.0
    ts: str
    cumulative_usd: float = 0.0
    tip_sha: str
    max_steps: int
    episodes: int
    note: str = ""
    calibration: Calibration | None = None


# `from __future__ import annotations` makes the `Calibration` above a string, and pydantic resolves
# it against the module in `sys.modules`. Running this file as a script registers it as `__main__`
# so that works, but loading it through `importlib` under any other name does not, and the failure
# is a confusing "not fully defined" the first time a ledger line is parsed. Resolving eagerly here
# means an analysis script can `import` this module and read a ledger without knowing that.
LedgerLine.model_rebuild()


class GridConfig(BaseModel):
    """Everything one invocation of this runner was told to do."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    arms: tuple[str, ...]
    grid_dir: Path
    model_dir: Path
    pool_file: Path
    traces_file: Path
    scenarios: int
    chunk_size: int
    episodes: int
    max_steps: int
    history_chars: int
    # Cells one execute_sweep runs at once. NOT a cohort pin and NOT in the plan identity (the
    # library excludes it on purpose): concurrency changes when a row is bought, never what it
    # measures, so raising it mid-grid re-buys nothing.
    concurrency: int
    cap_usd: float
    # Half-open chunk-index slice this process owns, or None for every chunk. The unit of
    # cross-process concurrency: chunk files are keyed by index and so are collision-free, so two
    # processes on disjoint ranges of ONE arm never touch the same file.
    chunk_range: tuple[int, int] | None
    only_models: tuple[str, ...]
    inject_fake_error: bool
    retry: bool
    smoke: bool


# ------------------------------------------------------------------------------------- primitives


def now_iso() -> str:
    """Current instant, ISO-8601 UTC, as every ledger line and artifact stamps it."""
    return datetime.now(tz=UTC).isoformat()


def tip_sha() -> str:
    """The main checkout's current commit, which is the cohort's identity.

    Refuses to guess: a grid whose cohort cannot be named is a grid whose rows cannot be compared
    with anything later, so an unreadable git state stops the run rather than stamping "unknown".
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GridStopped(
            f"cannot read the git tip of {REPO_ROOT}, so this run's cohort cannot be stamped: "
            f"{exc}. Run from the main checkout."
        ) from exc
    return result.stdout.strip()


def is_transient(error: str | None) -> bool:
    """Whether an unscored cell's error is worth exactly one more attempt."""
    return bool(error) and _TRANSIENT.search(error or "") is not None


def cell_key(outcome: ScenarioOutcome) -> str:
    """The identity of one cell, used for duplicate detection and the retried-once record."""
    return f"{outcome.scenario_id}|{outcome.model}|{outcome.episode}"


# --------------------------------------------------------------------------- ledger + cohort state


class GridState:
    """The grid directory's durable state: cohort pins, spend ledger, calibration, retry record."""

    def __init__(self, grid_dir: Path, cohort: Cohort) -> None:
        self._dir = grid_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self.cohort = self._reconcile_cohort(cohort)
        log.info("ledger to date: $%.4f already spent in this grid", self.spend_to_date())

    def _reconcile_cohort(self, cohort: Cohort) -> Cohort:
        """Adopt the recorded cohort, or refuse to append this run's cells to a different one."""
        path = self._dir / COHORT_FILE
        if not path.exists():
            path.write_text(cohort.model_dump_json(indent=2), encoding="utf-8")
            log.info(
                "stamped cohort %s (max_steps=%d, episodes=%d) in %s",
                cohort.tip_sha[:12],
                cohort.max_steps,
                cohort.episodes,
                self._dir,
            )
            return cohort
        recorded = Cohort.model_validate_json(path.read_text(encoding="utf-8"))
        difference = recorded.mismatch(cohort)
        if difference is not None:
            raise GridStopped(
                f"{self._dir} already holds cells from a different cohort ({difference}). Rows "
                "measured under different pins are not one matrix, so point --grid-dir at a NEW "
                "directory for this cohort, or check out the recorded tip "
                f"({recorded.tip_sha[:12]}) and re-run."
            )
        log.info("resuming cohort %s in %s", recorded.tip_sha[:12], self._dir)
        return recorded

    def pinned_pool_file(self, source: Path) -> Path:
        """The cohort's OWN copy of the candidate pool, made on first use and read forever after.

        `.wmo/pool.toml` is live local state that other lanes edit (it gained two candidates while
        this runner's own smoke was running). A grid that re-read it per chunk would silently
        change what it was measuring mid-cohort, and the merge would then refuse the arm for a
        pool-snapshot mismatch after paying for every chunk. Copying it once makes the roster part
        of the cohort, so an edit upstream is visible as a diff against this file instead of as a
        ruined arm.
        """
        pinned = self._dir / "pool.toml"
        if not pinned.exists():
            pinned.write_bytes(source.read_bytes())
            log.info("pinned the candidate pool for this cohort: %s -> %s", source, pinned)
        elif pinned.read_bytes() != source.read_bytes():
            log.warning(
                "%s has changed since this cohort pinned it; the grid keeps reading the pinned "
                "copy at %s (diff them if the new roster is what you want, then start a new "
                "cohort directory)",
                source,
                pinned,
            )
        return pinned

    def ledger_lines(self) -> list[LedgerLine]:
        """Every readable ledger line, re-read from disk.

        Re-read rather than cached, because the grid's spend is written by SEVERAL processes: one
        per arm, and optionally several per arm on disjoint chunk ranges. A cached total would only
        ever see this process's own cells, so each writer would believe the grid had spent a third
        of what it had.

        A line that does not parse is skipped with a warning instead of stopping the run: the one
        way to lose a line is a process dying mid-write, which costs one line of history, and
        refusing to read the rest would turn that into a lost budget state.
        """
        path = self._dir / LEDGER_FILE
        if not path.exists():
            return []
        lines: list[LedgerLine] = []
        for index, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            try:
                lines.append(LedgerLine.model_validate_json(raw))
            except ValueError:
                log.warning("ledger line %d of %s is unreadable; skipping it", index, path)
        return lines

    def spend_to_date(self) -> float:
        """The grid's cumulative measured spend, across every process, as of right now.

        Summed from the per-line halves rather than read off the last line's running total: with
        concurrent writers the last line's total is only that writer's view, and after a crash a
        truncated line would otherwise take the whole budget state with it.
        """
        return sum(line.candidate_usd + line.wm_usd for line in self.ledger_lines())

    def append(self, line: LedgerLine) -> None:
        """Append one ledger line atomically enough for several processes to share the file.

        One `open(..., "a")` per line and one `write` of a single newline-terminated JSON line. On
        every platform this runner targets, an append-mode write below the pipe buffer size does
        not interleave with another process's, so concurrent writers produce whole lines in some
        order rather than shredded ones. The stamped `cumulative_usd` is this writer's view at
        write time and is therefore ADVISORY under concurrency; the authoritative total is
        `spend_to_date`, which re-reads and sums the halves.
        """
        stamped = line.model_copy(
            update={"cumulative_usd": self.spend_to_date() + line.candidate_usd + line.wm_usd}
        )
        with (self._dir / LEDGER_FILE).open("a", encoding="utf-8") as handle:
            handle.write(stamped.model_dump_json() + "\n")

    def calibration(self) -> Calibration | None:
        """The persisted ratio match, if an earlier process in this grid already measured it."""
        path = self._dir / CALIBRATION_FILE
        if not path.exists():
            return None
        return Calibration.model_validate_json(path.read_text(encoding="utf-8"))

    def save_calibration(self, calibration: Calibration) -> None:
        (self._dir / CALIBRATION_FILE).write_text(
            calibration.model_dump_json(indent=2), encoding="utf-8"
        )

    def retried(self, arm: str, chunk: int) -> set[str]:
        """Cells this grid already spent a retry on for one chunk, so one-retry survives a restart.

        Keyed per CHUNK, not per arm. Two processes driving disjoint chunk ranges of the same arm
        would otherwise read-modify-write one shared file and each drop the other's entries, which
        would hand a still-failing cell a second and third retry. Per-chunk files are
        collision-free for the same reason the chunk matrices are: only one process owns an index.
        """
        path = self._dir / arm / RETRIED_FILE.format(chunk=chunk)
        if not path.exists():
            return set()
        keys = json.loads(path.read_text(encoding="utf-8"))
        return {str(key) for key in keys}

    def mark_retried(self, arm: str, chunk: int, keys: set[str]) -> None:
        """Record this chunk's spent retries before the retry runs, so a crash cannot lose them."""
        path = self._dir / arm / RETRIED_FILE.format(chunk=chunk)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sorted(keys), indent=2), encoding="utf-8")


# ------------------------------------------------------------------------------------ calibration


def render_transcript(trace: Trace, history_chars: int) -> str:
    """One trace as the user-role turn the candidate would be sent on its LAST step.

    The compressor only ever sees user-role content (`_CompressingProvider`), and that content is
    the agent's rendered turn: task, then every step so far with each observation clipped to
    `history_chars`. Reconstructed here in that shape, from the corpus, rather than measured on
    raw observation text, because the achieved ratio of a learned compressor depends on what the
    text looks like and the ratio has to be matched on what the grid will actually compress. The
    last turn is the largest, so this is the ratio at the transcript's full size.
    """
    lines = [f"TASK: {trace.steps[0].task or '(none)'}" if trace.steps else "TASK: (none)"]
    if trace.steps:
        lines.append("EPISODE SO FAR:")
    for index, step in enumerate(trace.steps):
        action = step.action
        if action.kind is ActionKind.TOOL_CALL:
            call = f"{action.name}({json.dumps(action.arguments, default=str)})"
        else:
            call = f"message: {action.content}"
        error_mark = " [ERROR]" if step.observation.is_error else ""
        lines.append(f"{index}. {call} -> {step.observation.content[:history_chars]}{error_mark}")
    lines.append("Your next move (JSON only):")
    return "\n".join(lines)


def keep_ratio(segments: list[str], config: CompressionConfig) -> float:
    """Achieved keep ratio of `config` over `segments`: compressed proxy tokens over raw."""
    compressor = get_compressor(config.compressor_id)
    result = compress_segments(compressor, segments, config)
    if result.tokens_in_raw == 0:
        raise GridStopped("the calibration sample carries no tokens; check --traces")
    return result.tokens_in_compressed / result.tokens_in_raw


def calibrate_truncate(
    segments: list[str], *, endpoint_config: CompressionConfig, sha: str
) -> Calibration:
    """Find truncate's dial whose ACHIEVED ratio matches the endpoint's on the same sample.

    Two passes over the dial (coarse then fine around the winner) because truncate drops whole
    words: its achieved ratio is a step function, so a uniform fine grid buys precision the steps
    do not have. The search is over the SAME segments the endpoint was measured on, which is the
    only way the two ratios are comparable.

    Raises:
        GridStopped: No dial value lands within `RATIO_TOLERANCE` of the endpoint's ratio.
    """
    raw_tokens = sum(estimate_tokens(segment) for segment in segments)
    endpoint_ratio = keep_ratio(segments, endpoint_config)
    log.info(
        "endpoint %s at aggressiveness %.2f keeps %.4f of %d proxy tokens over %d transcript(s)",
        endpoint_config.compressor_id,
        endpoint_config.aggressiveness,
        endpoint_ratio,
        raw_tokens,
        len(segments),
    )

    searched: list[tuple[float, float]] = []

    def measure(dial: float) -> float:
        ratio = keep_ratio(
            segments, CompressionConfig(compressor_id=TRUNCATE_ARM, aggressiveness=dial)
        )
        searched.append((round(dial, 4), ratio))
        log.info("  truncate aggressiveness %.3f -> achieved keep ratio %.4f", dial, ratio)
        return ratio

    coarse = [
        round(step * CALIBRATION_COARSE_STEP, 4)
        for step in range(int(1.0 / CALIBRATION_COARSE_STEP) + 1)
    ]
    best = min(coarse, key=lambda dial: abs(measure(dial) - endpoint_ratio))
    fine = [
        round(best + offset * CALIBRATION_FINE_STEP, 4)
        for offset in range(-4, 5)
        if 0.0 <= best + offset * CALIBRATION_FINE_STEP <= 1.0
    ]
    measured = {dial: ratio for dial, ratio in searched}
    for dial in fine:
        if dial not in measured:
            measured[dial] = measure(dial)
    chosen = min(measured, key=lambda dial: abs(measured[dial] - endpoint_ratio))
    gap = abs(measured[chosen] - endpoint_ratio)
    if gap > RATIO_TOLERANCE:
        raise GridStopped(
            f"no truncate aggressiveness matches the endpoint's achieved keep ratio "
            f"{endpoint_ratio:.4f} within +-{RATIO_TOLERANCE}: the closest is {chosen:g} at "
            f"{measured[chosen]:.4f} (gap {gap:.4f}). The control cannot be ratio-matched on this "
            "sample, so the compressed arms stop rather than measuring an unmatched control."
        )
    log.info(
        "calibrated truncate to aggressiveness %g (achieved %.4f vs endpoint %.4f, gap %.4f)",
        chosen,
        measured[chosen],
        endpoint_ratio,
        gap,
    )
    return Calibration(
        sample_size=len(segments),
        sample_tokens_raw=raw_tokens,
        endpoint_aggressiveness=endpoint_config.aggressiveness,
        endpoint_achieved_ratio=endpoint_ratio,
        searched=sorted(measured.items()),
        chosen_aggressiveness=chosen,
        chosen_achieved_ratio=measured[chosen],
        tolerance=RATIO_TOLERANCE,
        measured_at=now_iso(),
        tip_sha=sha,
    )


def calibration_sample(config: GridConfig, train_split: float, adapter: str) -> list[str]:
    """`CALIBRATION_SAMPLE` TRAIN-band transcripts, rendered as the turns the grid will compress.

    Sampled from the TRAIN band on purpose: calibration is a token-ratio measurement, but reading
    test-band text to set a knob that then scores the test band is the kind of contact this
    build's split discipline exists to avoid, and the train band answers the question just as well.
    """
    traces = get_adapter(adapter).from_file(str(config.traces_file))
    train, _band, _tiny = split_holdout(traces, train_split, (1.0 - train_split) / 2)
    with_steps = sorted((trace for trace in train if trace.steps), key=lambda t: t.trace_id)
    if not with_steps:
        raise GridStopped(
            f"{config.traces_file} has no train-band trace with steps, so the compressor's "
            "achieved ratio cannot be measured on realistic text"
        )
    picked = random.Random(CALIBRATION_SEED).sample(
        with_steps, min(CALIBRATION_SAMPLE, len(with_steps))
    )
    return [render_transcript(trace, config.history_chars) for trace in picked]


def endpoint_compression() -> CompressionConfig:
    """The endpoint arm's config, with the version of the implementation that will RUN.

    Building the compressor is the reachability check: the client verifies the live box's
    selection rule before it will attest append stability, so an unreachable or wrongly-deployed
    endpoint fails HERE, before any cell is paid for, instead of mid-grid.

    Raises:
        GridStopped: The endpoint is not usable, naming what to fix.
    """
    try:
        compressor = register_endpoint_compressor()
    except CompressorEndpointError as exc:
        raise GridStopped(
            f"the compressor endpoint is not usable, so the compressed arms stop: {exc} "
            "Nothing is substituted: an arm measured against a different compressor (or against "
            "raw text) is not the arm the grid claims."
        ) from exc
    return CompressionConfig(
        compressor_id=compressor.id,
        compressor_version=compressor.version,
        aggressiveness=ENDPOINT_AGGRESSIVENESS,
    )


def arm_compression(
    arm: str, config: GridConfig, state: GridState, *, train_split: float, adapter: str
) -> CompressionConfig | None:
    """The `CompressionConfig` an arm is measured under (None for the raw identity arm).

    The truncate arm's dial is read from the grid's persisted calibration when one exists, and
    measured (which needs the endpoint) when it does not. That is what lets `--arm truncate` run
    in its own process without re-deciding the control.
    """
    if arm == IDENTITY_ARM:
        return None
    if arm == ENDPOINT_ARM:
        return endpoint_compression()
    if arm == SCOPED_ENDPOINT_ARM:
        inner = endpoint_compression()  # the reachability check applies to the scoped arm too
        return CompressionConfig(
            compressor_id=SCOPED_ENDPOINT_ARM,
            compressor_version=get_compressor(SCOPED_ENDPOINT_ARM).version,
            aggressiveness=inner.aggressiveness,
        )
    existing = state.calibration()
    if existing is None:
        sample = calibration_sample(config, train_split, adapter)
        existing = calibrate_truncate(
            sample, endpoint_config=endpoint_compression(), sha=state.cohort.tip_sha
        )
        state.save_calibration(existing)
        state.append(
            LedgerLine(
                event="calibration",
                arm=TRUNCATE_ARM,
                ts=now_iso(),
                tip_sha=state.cohort.tip_sha,
                max_steps=config.max_steps,
                episodes=config.episodes,
                note=(
                    f"{existing.sample_size} train transcripts; endpoint keeps "
                    f"{existing.endpoint_achieved_ratio:.4f}, truncate "
                    f"{existing.chosen_aggressiveness:g} keeps {existing.chosen_achieved_ratio:.4f}"
                ),
                calibration=existing,
            )
        )
    else:
        log.info(
            "reusing this grid's calibration: truncate aggressiveness %g (achieved %.4f vs "
            "endpoint %.4f)",
            existing.chosen_aggressiveness,
            existing.chosen_achieved_ratio,
            existing.endpoint_achieved_ratio,
        )
    chosen_arm = SCOPED_TRUNCATE_ARM if arm == SCOPED_TRUNCATE_ARM else TRUNCATE_ARM
    return CompressionConfig(
        compressor_id=chosen_arm,
        compressor_version=get_compressor(chosen_arm).version,
        aggressiveness=existing.chosen_aggressiveness,
    )


# ------------------------------------------------------------------------------------ plan slicing


def scaled_cost_lines(lines: Sequence[CostLine], factor: float) -> tuple[CostLine, ...]:
    """The plan's projection rescaled to a slice of its scenarios.

    Exact rather than approximate: `plan_sweep` projects episodes, calls and dollars linearly in
    the scenario count, so scaling by the slice's share is the same arithmetic on fewer cells.
    """
    return tuple(
        line.model_copy(
            update={
                "episodes": round(line.episodes * factor),
                "calls": round(line.calls * factor),
                "usd": line.usd * factor,
            }
        )
        for line in lines
    )


def slice_plan(base: SweepPlan, scenarios: Sequence[Scenario], out_path: Path) -> SweepPlan:
    """`base` restricted to `scenarios`, writing to `out_path`.

    The scenario CUT is still the library's (`plan_sweep` sorted the held-out band by trace id and
    took the prefix); this only chooses how much of that cut one `execute_sweep` covers, so a
    chunk is a durability boundary and never a different measurement.
    """
    factor = len(scenarios) / len(base.scenarios) if base.scenarios else 0.0
    return base.model_copy(
        update={
            "scenarios": tuple(scenarios),
            "out_path": out_path,
            "cost_lines": scaled_cost_lines(base.cost_lines, factor),
        }
    )


def single_cell_plan(
    base: SweepPlan, pool: ModelPool, scenario: Scenario, out_path: Path
) -> SweepPlan:
    """`base` narrowed to ONE episode of one candidate on one scenario: the retry's unit of work.

    `episodes` is forced to 1 no matter what the arm runs at. A retry replaces a single failed row,
    so inheriting `--episodes 2` would buy two episodes and use one, and the discarded one would
    still be on the bill. The caller stamps the replacement with the failed row's own episode
    index, which is what keeps the (scenario, model, episode) key it is standing in for.
    """
    return base.model_copy(
        update={
            "pool": pool,
            "scenarios": (scenario,),
            "episodes": 1,
            "out_path": out_path,
            "cost_lines": scaled_cost_lines(
                [line for line in base.cost_lines if line.candidate == pool.models[0].name],
                1.0 / (len(base.scenarios) * base.episodes) if base.scenarios else 0.0,
            ),
        }
    )


# ------------------------------------------------------------------------------------- the arm run


class ArmRunner:
    """Runs one arm's chunks, retries its transient failures, merges its matrix."""

    def __init__(
        self,
        *,
        arm: str,
        config: GridConfig,
        state: GridState,
        base_plan: SweepPlan,
        world_model: WorldModel,
    ) -> None:
        self.arm = arm
        self.config = config
        self.state = state
        self.base_plan = base_plan
        self.world_model = world_model
        self.dir = config.grid_dir / arm
        self.dir.mkdir(parents=True, exist_ok=True)

    # -- chunks

    @property
    def chunk_count(self) -> int:
        """Chunks this arm actually has, from the cut the library produced.

        Derived from the cut rather than from `--scenarios`, because the two differ: scenarios are
        UNIQUE task prompts (`scenarios_from_traces` collapses a task repeated across traces into
        one), so a 45-scenario request against a corpus whose test band holds 20 distinct prompts
        is a 20-scenario grid. Deriving the count keeps the chunk indices contiguous, so a resume
        does not look for chunk files that were never possible.
        """
        cut = len(self.base_plan.scenarios)
        return -(-cut // self.config.chunk_size) if cut else 0

    def owned_chunks(self) -> list[int]:
        """The chunk indices THIS process runs: `--chunks A:B`, else every chunk of the arm.

        Only the running and retrying halves narrow. Merging always considers every chunk, so
        whichever process finishes last assembles the arm and the earlier ones say what is still
        missing.

        Raises:
            GridStopped: The requested range lies entirely outside the arm's chunks, which is
                almost always a launch-plan arithmetic error rather than an empty job.
        """
        if self.config.chunk_range is None:
            return list(range(self.chunk_count))
        begin, end = self.config.chunk_range
        owned = [index for index in range(self.chunk_count) if begin <= index < end]
        if not owned:
            raise GridStopped(
                f"--chunks {begin}:{end} selects no chunk of arm '{self.arm}', which has "
                f"{self.chunk_count} chunk(s) (indices 0..{max(self.chunk_count - 1, 0)}). Check "
                "the launch plan's ranges: the cut is smaller than a 45-scenario grid would be "
                "(see the scenario warning above)."
            )
        return owned

    def chunk_path(self, index: int) -> Path:
        return self.dir / f"chunk-{index}.json"

    def chunk_scenarios(self, index: int) -> tuple[Scenario, ...]:
        start = index * self.config.chunk_size
        return tuple(self.base_plan.scenarios[start : start + self.config.chunk_size])

    def loaded_chunk(self, index: int) -> OutcomeMatrix | None:
        """A chunk's matrix if it is on disk and loads clean, else None (so it will be re-run)."""
        path = self.chunk_path(index)
        if not path.exists():
            return None
        try:
            return OutcomeMatrix.load(path)
        except (OSError, ValueError) as exc:
            log.warning("%s exists but does not load (%s); re-running that chunk", path, exc)
            return None

    def forecast_usd(self, cells: int) -> float:
        """What the next `cells` are expected to cost, from THIS grid's measured cells.

        Measured, not projected: the plan's own estimate covers the candidate side only, and the
        calibration probe measured the world model's serve/judge side at roughly twice that on
        this corpus. Before any cell of this grid has been measured there is nothing to average,
        so the guard falls back to the probe's measured all-in figure.
        """
        paid = [
            line
            for line in self.state.ledger_lines()
            if line.event in {"chunk", "retry"} and line.cells
        ]
        measured_cells = sum(line.cells for line in paid)
        measured_usd = sum(line.candidate_usd + line.wm_usd for line in paid)
        per_cell = measured_usd / measured_cells if measured_cells else PROBE_USD_PER_CELL
        return per_cell * cells

    def run_chunks(self) -> None:
        """This process's chunks, skipping the ones already on disk, stopping at the cap."""
        for index in self.owned_chunks():
            scenarios = self.chunk_scenarios(index)
            if not scenarios:
                continue
            existing = self.loaded_chunk(index)
            if existing is not None:
                scored = sum(1 for outcome in existing.outcomes if outcome.scored)
                log.info(
                    "SKIP %s chunk %d: already on disk (%d cell(s), %d scored)",
                    self.arm,
                    index,
                    len(existing.outcomes),
                    scored,
                )
                self.state.append(
                    LedgerLine(
                        event="chunk-skipped",
                        arm=self.arm,
                        chunk=index,
                        cells=len(existing.outcomes),
                        scored=scored,
                        ts=now_iso(),
                        tip_sha=self.state.cohort.tip_sha,
                        max_steps=self.config.max_steps,
                        episodes=self.config.episodes,
                        note="resumed",
                    )
                )
                continue
            cells = len(self.base_plan.pool.models) * len(scenarios) * self.config.episodes
            forecast = self.forecast_usd(cells)
            # Re-read before every chunk, so a process sees what its SIBLINGS have spent. Under
            # concurrency the check is therefore approximate in one direction only: N processes can
            # each pass it and then each buy a chunk, so the grid may overshoot the cap by up to
            # one chunk per running process. Documented and accepted (a launch plan's worth of
            # chunks, not a launch plan's worth of arms); the alternative is a lock file that
            # would serialize the very concurrency this flag exists to buy.
            spent = self.state.spend_to_date()
            if spent + forecast > self.config.cap_usd:
                self.stop(
                    f"{self.arm} chunk {index} would take the grid past its "
                    f"${self.config.cap_usd:.0f} cap (spent ${spent:.2f} across all processes, "
                    f"this chunk forecast ${forecast:.2f} from measured cells)"
                )
            self.run_one_chunk(index, scenarios)

    def run_one_chunk(self, index: int, scenarios: tuple[Scenario, ...]) -> None:
        """Buy one chunk and persist it before anything else can go wrong."""
        plan = slice_plan(self.base_plan, scenarios, self.chunk_path(index))
        log.info(
            "%s chunk %d/%d: %d candidate(s) x %d scenario(s) x %d episode(s) = %d cell(s), "
            "projected candidate side $%.2f",
            self.arm,
            index,
            self.chunk_count - 1,
            len(plan.pool.models),
            len(scenarios),
            plan.episodes,
            plan.cells,
            plan.total_usd,
        )
        started = time.monotonic()
        run = self.execute(plan)
        wall = time.monotonic() - started
        scored = sum(1 for outcome in run.matrix.outcomes if outcome.scored)
        gap = run.metering_gap
        log.info(
            "%s chunk %d done in %.1fs: %d cell(s), %d scored, candidate $%.4f "
            "(compressor $%.4f), world model %s -> %s",
            self.arm,
            index,
            wall,
            len(run.matrix.outcomes),
            scored,
            run.candidate_usd,
            run.compressor_usd,
            f"${run.world_model_usd:.4f}" if gap is None else f"${run.world_model_usd:.4f} ({gap})",
            self.chunk_path(index),
        )
        self.state.append(
            LedgerLine(
                event="chunk",
                arm=self.arm,
                chunk=index,
                cells=len(run.matrix.outcomes),
                scored=scored,
                candidate_usd=run.candidate_usd,
                compressor_usd=run.compressor_usd,
                wm_usd=run.world_model_usd,
                wall_s=wall,
                ts=now_iso(),
                tip_sha=self.state.cohort.tip_sha,
                max_steps=self.config.max_steps,
                episodes=self.config.episodes,
                note=gap or "",
            )
        )
        spent = self.state.spend_to_date()
        if spent > self.config.cap_usd:
            self.stop(
                f"the ${self.config.cap_usd:.0f} cap is spent (${spent:.2f} measured all-in "
                "across every process in this grid)"
            )

    def execute(self, plan: SweepPlan) -> SweepRun:
        """One `execute_sweep`, with this arm's own runs directory for the world-model records."""
        world_model = self.world_model
        return execute_sweep(
            plan,
            world_model=world_model,
            env_factory=lambda: WorldModelEnv(world_model, score_on_close=True),
            on_outcome=lambda outcome: log.info(
                "  cell %s / %s ep%d: %s ($%.5f, %d step(s))%s",
                outcome.model,
                outcome.scenario_id,
                outcome.episode,
                "unscored" if outcome.reward is None else f"reward {outcome.reward:.3f}",
                outcome.cost_usd + outcome.compressor_cost_usd,
                outcome.steps,
                f" error={outcome.error}" if outcome.error else "",
            ),
            runs_dir=self.dir / "runs",
        )

    def stop(self, reason: str) -> None:
        """Record and raise a clean stop, with the command that resumes it."""
        self.state.append(
            LedgerLine(
                event="stop",
                arm=self.arm,
                ts=now_iso(),
                tip_sha=self.state.cohort.tip_sha,
                max_steps=self.config.max_steps,
                episodes=self.config.episodes,
                note=reason,
            )
        )
        chunks = (
            ""
            if self.config.chunk_range is None
            else f" --chunks {self.config.chunk_range[0]}:{self.config.chunk_range[1]}"
        )
        raise GridStopped(
            f"{reason}. Nothing was left half-written: every finished chunk is on disk. "
            "Resume (after raising --cap, if that is what stopped it) with:\n"
            f"  uv run python {Path(__file__).relative_to(REPO_ROOT)} --arm {self.arm}{chunks} "
            f"--episodes {self.config.episodes} --grid-dir {self.config.grid_dir}"
        )

    # -- retry pass

    def inject_fake_error(self) -> None:
        """Overwrite one measured row with a synthetic transient failure (SMOKE ONLY).

        The retry pass is the part of this runner a clean smoke would never exercise, and a code
        path that has never run is not proven. This deliberately destroys one real result in the
        smoke's throwaway grid so the retry can be watched end to end.
        """
        owned = self.owned_chunks()
        if not owned:
            return
        # The first chunk THIS process owns, so a `--chunks` slice that excludes 0 still exercises
        # the retry rather than silently skipping it.
        chunk = owned[0]
        matrix = self.loaded_chunk(chunk)
        if matrix is None or not matrix.outcomes:
            log.warning("no chunk %d row to inject a fake error into; retry unexercised", chunk)
            return
        target = matrix.outcomes[0]
        log.warning(
            "SMOKE: overwriting %s row with a synthetic InternalServerException to exercise the "
            "retry pass",
            cell_key(target),
        )
        matrix.outcomes[0] = target.model_copy(
            update={
                "reward": None,
                "success": False,
                "error": "InternalServerException: smoke-injected transient error",
            }
        )
        matrix.save(self.chunk_path(chunk))

    def retry_pass(self) -> None:
        """Re-execute the transiently-failed cells of THIS process's chunks, once each, in place.

        Scoped to the owned chunks for the same reason the runs are: two processes sharing an arm
        must not both decide to retry the same chunk's cells, which would double-buy them and race
        on the chunk file.
        """
        by_id = {scenario_id(scenario): scenario for scenario in self.base_plan.scenarios}
        for index in self.owned_chunks():
            matrix = self.loaded_chunk(index)
            if matrix is None:
                continue
            already = self.state.retried(self.arm, index)
            targets = [
                (position, outcome)
                for position, outcome in enumerate(matrix.outcomes)
                if not outcome.scored
                and is_transient(outcome.error)
                and cell_key(outcome) not in already
            ]
            if not targets:
                continue
            if len(targets) > len(matrix.outcomes) * SYSTEMIC_FAILURE_SHARE:
                # Most of a chunk failing transiently is not many independent blips, it is one
                # thing being down (a backend, the compressor box, the network). Retrying each
                # cell then buys the same failure N times at full episode price, so the chunk is
                # left as measured and the operator gets told what to fix before re-running it.
                log.warning(
                    "%s chunk %d: %d of %d cell(s) failed transiently, which reads as a systemic "
                    "outage rather than independent blips. NOT retrying them (that would buy the "
                    "same failure once per cell). First error: %s. Fix the backend, delete "
                    "%s, and re-run this arm.",
                    self.arm,
                    index,
                    len(targets),
                    len(matrix.outcomes),
                    targets[0][1].error,
                    self.chunk_path(index),
                )
                continue
            log.info(
                "%s chunk %d: retrying %d transiently-failed cell(s)", self.arm, index, len(targets)
            )
            for position, outcome in targets:
                already.add(cell_key(outcome))
                self.state.mark_retried(self.arm, index, already)
                replacement = self.retry_cell(index, outcome, by_id)
                if replacement is not None:
                    matrix.outcomes[position] = replacement
                    matrix.save(self.chunk_path(index))

    def retry_cell(
        self, chunk: int, outcome: ScenarioOutcome, by_id: dict[str, Scenario]
    ) -> ScenarioOutcome | None:
        """Buy one cell again. Returns the new row, or None to keep the honest error row."""
        scenario = by_id.get(outcome.scenario_id)
        if scenario is None:
            log.warning(
                "cannot retry %s: scenario %s is not in this run's cut",
                cell_key(outcome),
                outcome.scenario_id,
            )
            return None
        try:
            entry = self.base_plan.pool.entry(outcome.model)
        except KeyError as exc:
            log.warning("cannot retry %s: %s", cell_key(outcome), exc)
            return None
        out_path = self.dir / "retry" / f"chunk-{chunk}-{outcome.model}-{outcome.scenario_id}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plan = single_cell_plan(self.base_plan, ModelPool(models=[entry]), scenario, out_path)
        forecast = self.forecast_usd(1)
        spent = self.state.spend_to_date()
        if spent + forecast > self.config.cap_usd:
            self.stop(
                f"the retry of {cell_key(outcome)} would take the grid past its "
                f"${self.config.cap_usd:.0f} cap (spent ${spent:.2f} across all processes)"
            )
        started = time.monotonic()
        run = self.execute(plan)
        wall = time.monotonic() - started
        rows = run.matrix.outcomes
        if not rows:
            log.warning("retry of %s produced no row; keeping the error row", cell_key(outcome))
            return None
        fresh = rows[0].model_copy(update={"episode": outcome.episode})
        log.info(
            "retry %s: %s (%.1fs, candidate $%.4f)",
            cell_key(outcome),
            "still unscored" if fresh.reward is None else f"reward {fresh.reward:.3f}",
            wall,
            run.candidate_usd,
        )
        self.state.append(
            LedgerLine(
                event="retry",
                arm=self.arm,
                chunk=chunk,
                cells=1,
                scored=1 if fresh.scored else 0,
                candidate_usd=run.candidate_usd,
                compressor_usd=run.compressor_usd,
                wm_usd=run.world_model_usd,
                wall_s=wall,
                ts=now_iso(),
                tip_sha=self.state.cohort.tip_sha,
                max_steps=self.config.max_steps,
                episodes=self.config.episodes,
                note=f"{cell_key(outcome)}: {'recovered' if fresh.scored else 'still unscored'}",
            )
        )
        return fresh

    # -- merge

    def merge(self) -> Path | None:
        """Concatenate this arm's chunks into one matrix, or refuse and say what is missing.

        Three things are checked rather than assumed, because each one silently produces a matrix
        that reads as evidence and is not: a missing chunk (the arm is incomplete), a chunk swept
        against a different pool snapshot (its rows were chosen over different candidates at
        different prices), and a duplicated (scenario, model, episode) key (one cell counted
        twice, which reweights whatever is fitted from it).
        """
        matrices: list[tuple[int, OutcomeMatrix]] = []
        missing: list[int] = []
        for index in range(self.chunk_count):
            if not self.chunk_scenarios(index):
                continue
            matrix = self.loaded_chunk(index)
            if matrix is None:
                missing.append(index)
            else:
                matrices.append((index, matrix))
        if missing:
            log.warning(
                "%s is incomplete: chunk(s) %s are missing, so no merged matrix is written",
                self.arm,
                ", ".join(str(index) for index in missing),
            )
            return None
        reference = matrices[0][1]
        snapshot = [entry.model_dump(mode="json") for entry in reference.pool]
        outcomes: list[ScenarioOutcome] = []
        seen: dict[str, int] = {}
        for index, matrix in matrices:
            if [entry.model_dump(mode="json") for entry in matrix.pool] != snapshot:
                raise GridStopped(
                    f"{self.arm} chunk {index} was swept against a different pool snapshot than "
                    f"chunk {matrices[0][0]}, so its rows were chosen over different candidates "
                    "and the two cannot be one matrix. Re-run that chunk under the recorded pool, "
                    "or start a new cohort directory."
                )
            for outcome in matrix.outcomes:
                key = cell_key(outcome)
                if key in seen:
                    raise GridStopped(
                        f"{self.arm} has cell {key} in both chunk {seen[key]} and chunk {index}; a "
                        "duplicated cell would be counted twice by anything fitted from this "
                        "matrix. Delete the stale chunk file and re-run it."
                    )
                seen[key] = index
                outcomes.append(outcome)
        merged = OutcomeMatrix(pool=reference.pool, outcomes=outcomes)
        measured = merged.measured_compression()
        path = self.dir / MERGED_MATRIX
        merged.save(path)
        scored = sum(1 for outcome in merged.outcomes if outcome.scored)
        rows = coverage(merged)
        # The ratio match travels WITH the truncate arm's matrix: a control's number means nothing
        # without the achieved ratio it was matched to.
        calibration = self.state.calibration() if self.arm == TRUNCATE_ARM else None
        (self.dir / MERGED_META).write_text(
            json.dumps(
                {
                    "arm": self.arm,
                    "cohort": self.state.cohort.model_dump(),
                    "measured_compression": (
                        measured.model_dump() if measured is not None else None
                    ),
                    "chunks": len(matrices),
                    "cells": len(merged.outcomes),
                    "scored": scored,
                    "scenarios": len(merged.scenario_ids()),
                    "candidates": merged.model_names(),
                    "per_candidate_scored": {row.candidate: row.scored for row in rows},
                    "calibration": calibration.model_dump() if calibration else None,
                    "merged_at": now_iso(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        log.info(
            "merged %s: %d cell(s), %d scored, %d scenario(s), arm compression %s -> %s",
            self.arm,
            len(merged.outcomes),
            scored,
            len(merged.scenario_ids()),
            "raw" if measured is None else f"{measured.compressor_id}@{measured.aggressiveness:g}",
            path,
        )
        for row in rows:
            log.info(
                "  %-16s scored %3d unscored %3d%s",
                row.candidate,
                row.scored,
                row.unscored,
                f" first error: {row.first_error}" if row.first_error else "",
            )
        self.state.append(
            LedgerLine(
                event="merge",
                arm=self.arm,
                cells=len(merged.outcomes),
                scored=scored,
                ts=now_iso(),
                tip_sha=self.state.cohort.tip_sha,
                max_steps=self.config.max_steps,
                episodes=self.config.episodes,
                note=f"{len(matrices)} chunk(s) -> {path}",
            )
        )
        return path


# ------------------------------------------------------------------------------------------- main


def parse_chunk_range(spec: str | None) -> tuple[int, int] | None:
    """`A:B` as a half-open chunk-index range; None for every chunk.

    `A:` runs from A to the end and `:B` from the start, so a two-process split of one arm reads as
    `--chunks :2` and `--chunks 2:` without either side having to know the arm's chunk count.

    Raises:
        GridStopped: The spec is not `A:B`, or the range is empty or negative. A typo here silently
            costs a launch plan a whole range of chunks, so it is a hard stop rather than a warning.
    """
    if spec is None:
        return None
    if ":" not in spec:
        raise GridStopped(
            f"--chunks {spec!r} must be a half-open range 'A:B' (also 'A:' or ':B'), for example "
            "--chunks 0:2 for the first two chunks and --chunks 2: for the rest"
        )
    raw_begin, _, raw_end = spec.partition(":")
    try:
        begin = int(raw_begin) if raw_begin.strip() else 0
        end = int(raw_end) if raw_end.strip() else 1 << 30
    except ValueError as exc:
        raise GridStopped(f"--chunks {spec!r} has a non-integer bound: {exc}") from exc
    if begin < 0 or end <= begin:
        raise GridStopped(
            f"--chunks {spec!r} is empty or negative: the range is half-open, so A must be >= 0 "
            "and strictly less than B"
        )
    return (begin, end)


def build_config(args: argparse.Namespace) -> GridConfig:
    """Turn parsed arguments into the run's pins, applying the smoke's overrides."""
    smoke = bool(args.smoke)
    arms = (
        tuple(SMOKE_ARMS)
        if smoke and args.arm is None
        else (ALL_ARMS if args.arm is None else (args.arm,))
    )
    grid_dir = (
        Path(args.grid_dir) if args.grid_dir else (SMOKE_GRID_DIR if smoke else DEFAULT_GRID_DIR)
    )
    return GridConfig(
        arms=arms,
        grid_dir=grid_dir,
        model_dir=Path(args.model_dir),
        pool_file=Path(args.pool),
        traces_file=Path(args.traces),
        scenarios=args.scenarios
        if args.scenarios
        else (SMOKE_SCENARIOS if smoke else DEFAULT_SCENARIOS),
        chunk_size=args.chunk_size
        if args.chunk_size
        else (SMOKE_CHUNK if smoke else DEFAULT_CHUNK),
        episodes=args.episodes,
        max_steps=args.max_steps
        if args.max_steps
        else (SMOKE_MAX_STEPS if smoke else DEFAULT_MAX_STEPS),
        history_chars=args.history_chars,
        concurrency=args.concurrency,
        cap_usd=args.cap if args.cap else (SMOKE_CAP_USD if smoke else DEFAULT_CAP_USD),
        chunk_range=parse_chunk_range(args.chunks),
        only_models=tuple(args.only_model) if args.only_model else (SMOKE_MODELS if smoke else ()),
        inject_fake_error=smoke and not args.no_inject,
        retry=not args.no_retry,
        smoke=smoke,
    )


def restrict(pool: ModelPool, names: Sequence[str]) -> ModelPool:
    """The pool narrowed to `names`, in pool order, or the whole pool when `names` is empty."""
    if not names:
        return pool
    wanted = list(names)
    missing = [name for name in wanted if name not in {entry.name for entry in pool.models}]
    if missing:
        raise GridStopped(
            f"--only-model named {missing}, which the pool does not have; available: "
            f"{[entry.name for entry in pool.models]}"
        )
    return ModelPool(models=[entry for entry in pool.models if entry.name in set(wanted)])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """This runner's command line: what to run, where, and what it may spend."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--arm",
        choices=ALL_ARMS,
        default=None,
        help="Run ONE arm (so the three can run as staggered processes). Default: all three "
        "sequentially in this process.",
    )
    parser.add_argument(
        "--grid-dir", default=None, help=f"Grid directory (default {DEFAULT_GRID_DIR})."
    )
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Built world model.")
    parser.add_argument("--pool", default=str(DEFAULT_POOL), help="Candidate pool TOML.")
    parser.add_argument("--traces", default=str(DEFAULT_TRACES), help="Trace corpus.")
    parser.add_argument(
        "--scenarios",
        type=int,
        default=None,
        help=f"Test-band scenarios per arm (default {DEFAULT_SCENARIOS}).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help=f"Scenarios per chunk (default {DEFAULT_CHUNK}).",
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES, help="Episodes per cell.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Cells each chunk's execute_sweep runs at once (default 1, the sequential loop).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=f"Step budget per episode (default {DEFAULT_MAX_STEPS}).",
    )
    parser.add_argument(
        "--history-chars",
        type=int,
        default=DEFAULT_HISTORY_CHARS,
        help="Observation characters the agent sees on later turns.",
    )
    parser.add_argument(
        "--cap",
        type=float,
        default=None,
        help=f"Hard cumulative all-in cap for the whole grid (default ${DEFAULT_CAP_USD:.0f}).",
    )
    parser.add_argument(
        "--only-model",
        action="append",
        default=None,
        help="Restrict the pool to this candidate (repeatable).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Cheap end-to-end proof of this runner (see the module docstring).",
    )
    parser.add_argument(
        "--no-retry", action="store_true", help="Skip the transient-failure retry pass."
    )
    parser.add_argument(
        "--no-inject",
        action="store_true",
        help="In --smoke, do not inject a synthetic transient failure.",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Merge whatever chunks are on disk and exit, spending nothing.",
    )
    parser.add_argument(
        "--chunks",
        default=None,
        help="Half-open chunk-index range this process runs, e.g. 0:2 or 2: . Lets several OS "
        "processes drive ONE arm on disjoint ranges (chunk files are keyed by index, so they never "
        "collide). Default: every chunk of the arm. Merging always considers every chunk, so "
        "whichever process finishes last assembles the arm.",
    )
    parser.add_argument(
        "--env-file",
        action="append",
        default=None,
        help="Extra KEY=VALUE file layered over the main checkout's .env, for credentials kept "
        "elsewhere (the compressor endpoint's WMO_COMPRESSOR_* live in the compression lane's "
        ".env). Already-set variables are never overridden. Repeatable.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = parse_args(argv)
    # The main checkout's .env first, then any layered file, and `load_env_file` never overrides
    # what is already set: the compressor endpoint's URL, token and pinned certificate are kept
    # in the compression lane's own gitignored .env, so --env-file points at them instead of the
    # credentials being copied between checkouts.
    load_env_file(REPO_ROOT / ".env")
    for extra in args.env_file or []:
        path = Path(extra)
        if not path.is_file():
            raise GridStopped(f"--env-file {path} is not a file")
        load_env_file(path)
    config = build_config(args)
    sha = tip_sha()

    harness_config = resolve_config(config.model_dir)
    # The cohort is settled before the pool is read, because the pool the grid measures IS the
    # cohort's pinned copy rather than whatever `.wmo/pool.toml` says at this moment.
    cohort = Cohort(
        tip_sha=sha,
        max_steps=config.max_steps,
        episodes=config.episodes,
        scenarios=config.scenarios,
        chunk_size=config.chunk_size,
        history_chars=config.history_chars,
        model_dir=str(config.model_dir),
        pool_file=str(config.grid_dir / "pool.toml"),
        traces_file=str(config.traces_file),
        created=now_iso(),
    )
    state = GridState(config.grid_dir, cohort)
    # ZERO-EPISODE PREFLIGHT over the WHOLE pinned roster, before a cell is bought and even when
    # --only-model narrows what this process will run: `preflight_pool` builds every candidate's
    # SDK client and resolves its credentials locally, without a single request. So a bad entry is
    # a boundary error here rather than a mid-grid abort with earlier candidates already paid for,
    # and the smoke checks all 11 candidates while spending on 2.
    preflight = preflight_pool(state.pinned_pool_file(config.pool_file))
    log.info(
        "preflight: all %d pinned candidate(s) resolved without a request (%s)",
        len(preflight.pool.models),
        ", ".join(entry.name for entry in preflight.pool.models),
    )
    pool = restrict(preflight.pool, config.only_models)
    for risk in preflight.deferred:
        if risk.candidate in {entry.name for entry in pool.models}:
            log.info(
                "first-cell risk for %s (kind=%s): %s", risk.candidate, risk.kind.value, risk.risk
            )

    log.info(
        "grid: arms=%s, chunks=%s, up to %d scenario(s) in chunks of %d, %d candidate(s), "
        "max_steps=%d, episodes=%d, cap $%.0f, dir %s",
        ",".join(config.arms),
        "all" if config.chunk_range is None else f"{config.chunk_range[0]}:{config.chunk_range[1]}",
        config.scenarios,
        config.chunk_size,
        len(pool.models),
        config.max_steps,
        config.episodes,
        config.cap_usd,
        config.grid_dir,
    )

    world_model, _serve_provider = load_world_model(config.model_dir)

    for arm in config.arms:
        compression = (
            None
            if args.merge_only
            else arm_compression(
                arm,
                config,
                state,
                train_split=harness_config.train_split,
                adapter=harness_config.trace_adapter,
            )
        )
        base_plan = plan_sweep(
            model_dir=config.model_dir,
            config=harness_config,
            pool=pool,
            out_path=config.grid_dir / arm / MERGED_MATRIX,
            traces_file=config.traces_file,
            scenarios=config.scenarios,
            episodes=config.episodes,
            max_steps=config.max_steps,
            assume_input_tokens=2000,
            assume_output_tokens=250,
            history_chars=config.history_chars,
            compression=compression,
            max_concurrency=config.concurrency,
        )
        runner = ArmRunner(
            arm=arm,
            config=config,
            state=state,
            base_plan=base_plan,
            world_model=world_model,
        )
        if len(base_plan.scenarios) < config.scenarios:
            # Said out loud rather than absorbed: the arm is smaller than it was asked for, which
            # changes the grid's power, and a runner that quietly measured fewer scenarios than the
            # registered design would let the shortfall reach the writeup unnoticed.
            log.warning(
                "%s: asked for %d scenario(s) but the test band has only %d DISTINCT task "
                "prompt(s), so this arm is %d scenario(s) in %d chunk(s) (%d cell(s)). Raise "
                "--episodes to recover power; the cut cannot be padded.",
                arm,
                config.scenarios,
                len(base_plan.scenarios),
                len(base_plan.scenarios),
                runner.chunk_count,
                base_plan.cells,
            )
        if not args.merge_only:
            log.info(
                "=== arm %s (%s) ===",
                arm,
                "raw text (no compression)"
                if compression is None
                else f"{compression.compressor_id} v{compression.compressor_version} "
                f"@ {compression.aggressiveness:g}",
            )
            runner.run_chunks()
            if config.inject_fake_error:
                runner.inject_fake_error()
            if config.retry:
                runner.retry_pass()
        runner.merge()

    log.info(
        "grid spend so far: $%.4f all-in across every process (cap $%.0f)",
        state.spend_to_date(),
        config.cap_usd,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GridStopped, SweepError) as error:
        logging.getLogger("tau-grid").error("%s", error)
        raise SystemExit(2) from error
