"""One-turn agentic proposal of complete harness source trees from a scored population.

`ProjectCandidateProposer` implements the population loop's `CandidateProposer` seam with the
paper's protocol: a FRESH agent project (E2B sandbox) per proposal slot, a navigable filesystem
history of every evaluated candidate (complete source, score report, and the raw per-trial
`wmh-run.json` transcripts plus verifier outputs), a fixed-seed stage in `candidate/`, exactly
one agent turn with `retry_recoverable=False`, then a bounded snapshot, parse, and `node`
interface validation. Every failure path (agent error, missing submit, snapshot failure,
invalid tree, syntax error, or transport death) consumes the slot as a
:class:`CandidateProposalError` with raw evidence persisted under `proposals/slot-NNNN/` in the
run directory, so later slots learn from it.
"""

from __future__ import annotations

import json
import logging
import shlex
from collections.abc import Callable, Collection, Sequence
from pathlib import Path
from typing import Protocol

from pydantic import JsonValue

from wmh.agents.project import (
    DEFAULT_SOURCE_TREE_MAX_BYTES,
    DEFAULT_SOURCE_TREE_MAX_FILES,
    AgentProjectRun,
    ProjectBashResult,
)
from wmh.core.text import validate_durable_text
from wmh.harness.doc import HarnessDoc
from wmh.harness.live_session import SessionEvent
from wmh.harness.population import (
    CandidateProposal,
    CandidateProposalError,
    EvaluatedCandidate,
    candidate_slot_id,
)
from wmh.harness.runtime import HarnessSearchCancelled
from wmh.harness.scoring import ScoreCell
from wmh.harness.source_tree import HarnessSourceTree
from wmh.providers.base import ToolCallingProvider

logger = logging.getLogger(__name__)

CANDIDATE_STAGE_DIR = "candidate"
PROPOSALS_DIR = "proposals"
# Upper bound on a feedback directive threaded into every proposal request.
MAX_DIRECTIVE_CHARS = 20_000
# Per-candidate budget for materialized raw trial evidence; breaches truncate, never halt.
DEFAULT_CANDIDATE_HISTORY_BYTES = 256 * 1024 * 1024
# Per-file cap before head/tail truncation (transcripts keep both ends plus a marker).
DEFAULT_HISTORY_FILE_BYTES = 2 * 1024 * 1024
_WMH_RUN_FILENAME = "wmh-run.json"
_TRIAL_AGENT_DIR = "agent"
_TRIAL_VERIFIER_DIR = "verifier"


class CandidateProject(Protocol):
    """The project-side operations one proposal slot needs (a fresh instance per slot)."""

    workspace: str

    def write_text(self, path: str, content: str) -> None: ...

    def run_bash(self, command: str) -> ProjectBashResult: ...

    def stage_source_tree(self, tree: HarnessSourceTree, dest: str) -> None: ...

    def snapshot_source_tree(
        self,
        directory: str,
        *,
        max_files: int,
        max_bytes: int,
    ) -> HarnessSourceTree: ...

    def run(
        self,
        agent: HarnessDoc,
        provider: ToolCallingProvider,
        instruction: str,
        *,
        on_event: Callable[[SessionEvent], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        writable_files: Collection[str] | None = None,
        retry_recoverable: bool = True,
    ) -> AgentProjectRun: ...

    def close(self) -> None: ...


class ProjectCandidateProposer:
    """Run one contained coding turn per slot against the complete scored population."""

    def __init__(
        self,
        agent: HarnessDoc,
        provider: ToolCallingProvider,
        *,
        project_factory: Callable[[], CandidateProject],
        run_dir: str | Path,
        directive: str = "",
        on_event: Callable[[SessionEvent], None] | None = None,
        max_source_files: int = DEFAULT_SOURCE_TREE_MAX_FILES,
        max_source_bytes: int = DEFAULT_SOURCE_TREE_MAX_BYTES,
        max_candidate_history_bytes: int = DEFAULT_CANDIDATE_HISTORY_BYTES,
        max_history_file_bytes: int = DEFAULT_HISTORY_FILE_BYTES,
    ) -> None:
        for field, value in (
            ("max_source_files", max_source_files),
            ("max_source_bytes", max_source_bytes),
            ("max_candidate_history_bytes", max_candidate_history_bytes),
            ("max_history_file_bytes", max_history_file_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if directive:
            validate_durable_text(directive, field="proposal directive")
            if len(directive) > MAX_DIRECTIVE_CHARS:
                raise ValueError(f"proposal directive exceeds {MAX_DIRECTIVE_CHARS} characters")
        self._agent = agent
        self._provider = provider
        self._project_factory = project_factory
        self._run_dir = Path(run_dir)
        self._directive = directive
        self._on_event = on_event
        self._max_source_files = max_source_files
        self._max_source_bytes = max_source_bytes
        self._max_candidate_history_bytes = max_candidate_history_bytes
        self._max_history_file_bytes = max_history_file_bytes

    def propose(
        self,
        population: Sequence[EvaluatedCandidate],
        *,
        slot: int,
        should_cancel: Callable[[], bool] | None = None,
    ) -> CandidateProposal:
        """Produce one new candidate from the fixed seed stage, never a selected parent."""
        if not population:
            raise ValueError("candidate proposal requires an evaluated seed in the population")
        if isinstance(slot, bool) or not isinstance(slot, int) or slot < 1:
            raise ValueError("slot must be a positive integer")
        _check_cancelled(should_cancel)
        candidate_id = candidate_slot_id(slot)
        slot_dir = self._run_dir / PROPOSALS_DIR / f"slot-{slot:04d}"
        # A crashed earlier attempt at this same slot is still evidence: move it aside into
        # attempt-K/ (never delete it) so the redone turn starts clean but later slots and
        # humans can still read what happened.
        _set_aside_prior_attempt(slot_dir)
        slot_dir.mkdir(parents=True, exist_ok=True)
        try:
            return self._propose(
                tuple(population),
                candidate_id=candidate_id,
                slot=slot,
                slot_dir=slot_dir,
                should_cancel=should_cancel,
            )
        except (HarnessSearchCancelled, CandidateProposalError):
            raise
        except Exception as error:
            # Transport and materialization failures consume the paid slot (never a silent
            # replay of a proposal turn); the population loop records the evidence path.
            reason = f"proposal infrastructure failed: {error}"
            _write_json(slot_dir / "error.json", {"candidate_id": candidate_id, "reason": reason})
            raise CandidateProposalError(
                candidate_id, reason, evidence_dir=str(slot_dir)
            ) from error

    def _propose(
        self,
        population: tuple[EvaluatedCandidate, ...],
        *,
        candidate_id: str,
        slot: int,
        slot_dir: Path,
        should_cancel: Callable[[], bool] | None,
    ) -> CandidateProposal:
        project = self._project_factory()
        try:
            self._materialize_history(project, population)
            self._materialize_prior_proposals(project, slot)
            project.stage_source_tree(population[0].source, CANDIDATE_STAGE_DIR)
            request = _proposal_request(
                candidate_id=candidate_id,
                stage_dir=f"{project.workspace}/{CANDIDATE_STAGE_DIR}",
                slot=slot,
                population_count=len(population),
                directive=self._directive,
            )
            (slot_dir / "REQUEST.md").write_text(request, encoding="utf-8")
            project.write_text(f"{PROPOSALS_DIR}/slot-{slot:04d}/REQUEST.md", request)
            _check_cancelled(should_cancel)

            events: list[SessionEvent] = []

            def sink(event: SessionEvent) -> None:
                events.append(event)
                if self._on_event is not None:
                    try:
                        self._on_event(event)
                    except Exception:  # noqa: BLE001 - a sink error must never abort the proposal
                        logger.warning("proposer on_event sink raised", exc_info=True)

            run_error: str | None = None
            try:
                project.run(
                    self._agent,
                    self._provider,
                    request,
                    on_event=sink,
                    should_cancel=should_cancel,
                    writable_files=(),
                    retry_recoverable=False,
                )
            except HarnessSearchCancelled:
                self._write_events(slot_dir, events)
                raise
            except Exception as error:  # noqa: BLE001 - the failed turn is slot evidence
                run_error = str(error)
            self._write_events(slot_dir, events)
            _check_cancelled(should_cancel)

            source: HarnessSourceTree | None = None
            snapshot_error: str | None = None
            try:
                source = project.snapshot_source_tree(
                    CANDIDATE_STAGE_DIR,
                    max_files=self._max_source_files,
                    max_bytes=self._max_source_bytes,
                )
            except Exception as error:  # noqa: BLE001 - a bad snapshot is slot evidence
                snapshot_error = str(error)
            if source is not None:
                for item in source.files:
                    target = slot_dir / "source" / item.path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    # Exact bytes: newline translation would silently change content hashes.
                    target.write_bytes(item.content.encode("utf-8"))

            failures: list[str] = []
            if run_error is not None:
                failures.append(f"agent turn failed: {run_error}")
            if not any(event.kind == "submit" for event in events):
                failures.append("agent turn did not submit a completed candidate")
            if snapshot_error is not None:
                failures.append(f"candidate snapshot failed: {snapshot_error}")
            candidate: HarnessDoc | None = None
            if source is not None:
                try:
                    candidate = source.to_doc(candidate_id)
                except (TypeError, ValueError) as error:
                    failures.append(f"candidate source is not a valid harness: {error}")
            if source is not None and candidate is not None and not failures:
                failures.extend(_interface_errors(project, source))

            _write_json(
                slot_dir / "status.json",
                {
                    "candidate_id": candidate_id,
                    "valid": candidate is not None and not failures,
                    "candidate_doc_hash": None if candidate is None else candidate.doc_hash,
                    "source_tree_hash": None if source is None else source.tree_hash,
                    "errors": failures,
                },
            )
            if failures or source is None or candidate is None:
                reason = "; ".join(failures) or "candidate source was not captured"
                raise CandidateProposalError(candidate_id, reason, evidence_dir=str(slot_dir))
            return CandidateProposal(candidate_id=candidate_id, source=source, candidate=candidate)
        finally:
            try:
                project.close()
            except Exception:  # noqa: BLE001 - closing must not mask the slot outcome
                logger.warning(
                    "proposer project close failed; the sandbox may stay billable until its "
                    "lifetime timeout",
                    exc_info=True,
                )

    def _materialize_history(
        self,
        project: CandidateProject,
        population: tuple[EvaluatedCandidate, ...],
    ) -> None:
        """Write every evaluated candidate's source, report, and raw trial evidence."""
        manifest: list[dict[str, JsonValue]] = []
        for evaluated in population:
            directory = f"history/{evaluated.candidate_id}"
            for item in evaluated.source.files:
                project.write_text(f"{directory}/source/{item.path}", item.content)
            report = evaluated.report
            cells: list[dict[str, JsonValue]] = []
            budget = self._max_candidate_history_bytes
            for cell in report.cells:
                trial_dir = f"{directory}/trials/{cell.task_id}/attempt-{cell.attempt}"
                cells.append(
                    {
                        "task_id": cell.task_id,
                        "attempt": cell.attempt,
                        "reward": cell.reward,
                        "passed": cell.passed,
                        "note": cell.note,
                        "trial_dir": trial_dir,
                    }
                )
                budget = self._copy_trial_evidence(project, trial_dir, cell, budget)
            project.write_text(
                f"{directory}/report.json",
                _json(
                    {
                        "candidate_id": evaluated.candidate_id,
                        "score": report.score,
                        "pass_rate": report.pass_rate,
                        "reward_mode": report.reward_mode,
                        "attempts": report.request.attempts,
                        "cells": cells,
                    }
                ),
            )
            manifest.append(
                {
                    "candidate_id": evaluated.candidate_id,
                    "score": report.score,
                    "pass_rate": report.pass_rate,
                    "by_task": {
                        task_id: sum(1 for cell in cells_ if cell.passed) / len(cells_)
                        for task_id, cells_ in report.by_task().items()
                    },
                    "source_dir": f"{directory}/source",
                    "report": f"{directory}/report.json",
                    "trials_dir": f"{directory}/trials",
                }
            )
        project.write_text(
            "history/manifest.json",
            _json(
                {
                    "candidate_count": len(manifest),
                    "candidates": manifest,
                    "proposals_dir": PROPOSALS_DIR,
                }
            ),
        )

    def _copy_trial_evidence(
        self,
        project: CandidateProject,
        trial_dir: str,
        cell: ScoreCell,
        budget: int,
    ) -> int:
        """Copy one trial's transcript and verifier output (never harbor's ceremony files).

        Every gap is marked where the proposer will look: a trial with no recorded artifact
        directory gets `NO-EVIDENCE.md`, and a trial whose files were truncated or omitted by
        the byte budget gets `EVIDENCE-TRUNCATED.md` naming exactly what is incomplete.
        """
        if not cell.artifact_dir:
            project.write_text(
                f"{trial_dir}/NO-EVIDENCE.md",
                "This trial recorded no artifact directory; no raw evidence was captured.",
            )
            return budget
        artifact_dir = Path(cell.artifact_dir)
        sources: list[tuple[Path, str]] = []
        transcript = artifact_dir / _TRIAL_AGENT_DIR / _WMH_RUN_FILENAME
        if transcript.is_file():
            sources.append((transcript, f"{trial_dir}/{_WMH_RUN_FILENAME}"))
        verifier_dir = artifact_dir / _TRIAL_VERIFIER_DIR
        if verifier_dir.is_dir():
            sources.extend(
                (path, f"{trial_dir}/{_TRIAL_VERIFIER_DIR}/{path.relative_to(verifier_dir)}")
                for path in sorted(verifier_dir.rglob("*"))
                if path.is_file()
            )
        incomplete: list[str] = []
        for host_path, target in sources:
            if budget <= 0:
                incomplete.append(f"omitted (byte budget reached): {target}")
                continue
            try:
                content, was_truncated = _read_evidence(host_path, self._max_history_file_bytes)
            except OSError as error:
                content, was_truncated = f"[unreadable evidence file: {error}]", False
            project.write_text(target, content)
            budget -= len(content.encode("utf-8"))
            if was_truncated:
                incomplete.append(f"head/tail truncated: {target}")
        if incomplete:
            project.write_text(
                f"{trial_dir}/EVIDENCE-TRUNCATED.md",
                "This trial's raw evidence is incomplete:\n" + "\n".join(incomplete),
            )
        return budget

    def _materialize_prior_proposals(self, project: CandidateProject, slot: int) -> None:
        """Upload every earlier slot's proposal trace so failures teach later turns."""
        proposals_root = self._run_dir / PROPOSALS_DIR
        if not proposals_root.is_dir():
            return
        budget = self._max_candidate_history_bytes
        omitted: list[str] = []
        for slot_dir in sorted(proposals_root.iterdir()):
            if not slot_dir.is_dir() or slot_dir.name == f"slot-{slot:04d}":
                continue
            for path in sorted(slot_dir.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(proposals_root).as_posix()
                if budget <= 0:
                    omitted.append(relative)
                    continue
                try:
                    content, _was_truncated = _read_evidence(path, self._max_history_file_bytes)
                except OSError as error:
                    content = f"[unreadable evidence file: {error}]"
                project.write_text(f"{PROPOSALS_DIR}/{relative}", content)
                budget -= len(content.encode("utf-8"))
        if omitted:
            project.write_text(
                f"{PROPOSALS_DIR}/TRUNCATED.md",
                "Prior proposal evidence beyond the byte budget was omitted:\n"
                + "\n".join(omitted),
            )

    def _write_events(self, slot_dir: Path, events: Sequence[SessionEvent]) -> None:
        _write_json(
            slot_dir / "events.json",
            [{"kind": event.kind, "payload": event.payload} for event in events],
        )


# In-sandbox syntax validation for candidate .ts/.js files. `node --check` cannot be used for
# TypeScript: it does NOT strip types under --check (verified on node 22.23), so it falsely
# rejects valid TS including the seed's own vendored files. `stripTypeScriptTypes` (node >=
# 22.13) fully parses TS module grammar and throws on syntax errors, and plain JS is a subset
# of that grammar, so one code path validates both. When the sandbox's node predates it, the
# script reports the skip marker and validation is SKIPPED rather than false-rejecting every
# candidate.
_TS_VALIDATION_SKIP_MARKER = "typescript-validation-skipped"
_TS_CHECK_SCRIPT = f"""\
const {{ readFileSync }} = require('node:fs');
let strip;
try {{
  ({{ stripTypeScriptTypes: strip }} = require('node:module'));
}} catch {{}}
if (typeof strip !== 'function') {{
  console.log('{_TS_VALIDATION_SKIP_MARKER}: node:module.stripTypeScriptTypes unavailable');
  process.exit(0);
}}
let failed = false;
for (const path of process.argv.slice(1)) {{
  try {{
    strip(readFileSync(path, 'utf8'));
  }} catch (error) {{
    failed = true;
    console.error(path + ': ' + (error && error.message ? error.message : String(error)));
  }}
}}
process.exit(failed ? 1 : 0);
"""


def _interface_errors(project: CandidateProject, source: HarnessSourceTree) -> list[str]:
    """The paper's interface-validation gate: parse-check every candidate code file.

    A candidate that cannot parse must never burn a paid evaluation. One in-sandbox node
    invocation strips/parses every `.ts`/`.js` file via `node:module.stripTypeScriptTypes`
    (which throws on bad syntax); failures preserve node's per-file error as slot evidence.
    """
    staged = [
        f"{CANDIDATE_STAGE_DIR}/{item.path}"
        for item in source.files
        if item.path.endswith((".ts", ".js"))
    ]
    if not staged:
        return []
    quoted_paths = " ".join(shlex.quote(path) for path in staged)
    result = project.run_bash(f"node -e {shlex.quote(_TS_CHECK_SCRIPT)} {quoted_paths}")
    if result.exit_code == 0:
        if _TS_VALIDATION_SKIP_MARKER in result.stdout:
            logger.warning("candidate interface validation skipped: %s", result.stdout.strip())
        return []
    detail = result.stderr or result.stdout or f"exit {result.exit_code}"
    return [f"interface validation failed: {detail}"]


def _set_aside_prior_attempt(slot_dir: Path) -> None:
    """Move a crashed prior attempt's evidence into `attempt-K/` instead of deleting it."""
    if not slot_dir.is_dir():
        return
    prior = [path for path in slot_dir.iterdir() if not path.name.startswith("attempt-")]
    if not prior:
        return
    attempt_count = sum(
        1 for path in slot_dir.iterdir() if path.is_dir() and path.name.startswith("attempt-")
    )
    attempt_dir = slot_dir / f"attempt-{attempt_count + 1}"
    attempt_dir.mkdir()
    for path in prior:
        path.rename(attempt_dir / path.name)


def _read_evidence(path: Path, max_bytes: int) -> tuple[str, bool]:
    """Read one evidence file, head/tail-truncating oversized content with a marker.

    Returns the content and whether it was truncated.
    """
    size = path.stat().st_size
    if size <= max_bytes:
        return path.read_bytes().decode("utf-8", errors="replace"), False
    half = max_bytes // 2
    with path.open("rb") as handle:
        head = handle.read(half)
        handle.seek(size - half)
        tail = handle.read(half)
    return (
        head.decode("utf-8", errors="replace")
        + f"\n... {size - max_bytes} bytes truncated ...\n"
        + tail.decode("utf-8", errors="replace")
    ), True


def _proposal_request(
    *,
    candidate_id: str,
    stage_dir: str,
    slot: int,
    population_count: int,
    directive: str = "",
) -> str:
    previous_trace = (
        f"The most recent proposal-turn trace is `{PROPOSALS_DIR}/slot-{slot - 1:04d}/`; "
        "failed turns keep their captured source and errors in the same layout."
        if slot > 1
        else "There is no earlier proposal turn in this project."
    )
    directive_block = (
        "The user directing this optimization gave the following feedback. Treat it as the "
        f"primary objective of this candidate:\n\n{directive}\n\n"
        if directive
        else ""
    )
    return f"""Produce exactly one complete harness candidate: {candidate_id}.

Read `history/manifest.json`. It indexes all {population_count} evaluated candidates: complete
source directories, full score reports, and raw per-trial evidence under
`history/<candidate>/trials/<task>/attempt-N/` (`{_WMH_RUN_FILENAME}` is the complete agent
transcript for that trial; `verifier/` holds the verifier's own output). Earlier proposal-turn
traces, including failed ones, remain under `{PROPOSALS_DIR}/`. {previous_trace} Use the full
population as evidence.

{directive_block}Your only candidate output is this directory:
`{stage_dir}`

The host initialized it with a complete, freely editable copy of the fixed first evaluated seed at
`history/candidate-0000/source`. Every proposal slot receives that same fixed scaffold regardless
of later candidates or scores. Use bash to inspect the immutable project evidence and to create,
edit, delete, replace, and test files inside the candidate directory. Leave one complete
standalone harness source tree there; it must not import from or otherwise depend on `history/`,
`{PROPOSALS_DIR}/`, or any other project path. The final portable source tree must contain a
UTF-8 `SYSTEM.md`; optional `config.toml`, `runtime.py`, `skills/*.md`, and code files must
remain valid in the portable source format and parse together as one complete harness.

Filename grammar (strict): outside the reserved names above, every file path must be lowercase
kebab-case, meaning runs of [a-z0-9] separated by single '/', '.', or '-' characters (for example
`src/agent-loop.ts`). Uppercase letters and underscores are rejected (`src/a_b.ts` is invalid).
Because '/' and '.' both canonicalize to '-', paths like `a-b.ts` and `a/b.ts` collide; paths
differing only by letter case collide too, and no file path may also be a directory prefix of
another (`a` next to `a/b.ts`). One bad filename invalidates the whole candidate, so name files
carefully instead of spending this slot on cosmetics.

Do not modify immutable project paths. When the candidate is complete, call submit. The host
snapshots the directory once after this turn, parse-checks every `.ts`/`.js` file with node's
TypeScript syntax stripper (a syntax error anywhere invalidates the candidate), and will not ask
for a repair.

This turn's immutable request and trace are stored under `{PROPOSALS_DIR}/slot-{slot:04d}/`.
"""


def _json(value: JsonValue) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _write_json(path: Path, value: JsonValue) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json(value) + "\n", encoding="utf-8")


def _check_cancelled(should_cancel: Callable[[], bool] | None) -> None:
    if should_cancel is not None and should_cancel():
        raise HarnessSearchCancelled("harness search cancelled")
