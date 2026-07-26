"""Tests for the fresh-project-per-slot proposer (in-memory fake project, no E2B)."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path

import pytest
from llm_waterfall import ChatResponse

from wmh.agents.default import default_agent
from wmh.agents.optimizer import optimizer_agent
from wmh.agents.project import AgentProjectRun, ProjectBashResult
from wmh.harness import project_proposer
from wmh.harness.doc import HarnessDoc
from wmh.harness.live_session import SessionEvent
from wmh.harness.population import CandidateProposalError, EvaluatedCandidate
from wmh.harness.project_proposer import (
    DEFAULT_CANDIDATE_HISTORY_BYTES,
    DEFAULT_HISTORY_FILE_BYTES,
    ProjectCandidateProposer,
)
from wmh.harness.runtime import TokenUsage
from wmh.harness.scoring import ScoreCell, ScoreReport, ScoreRequest
from wmh.harness.source_tree import HarnessSourceFile, HarnessSourceTree
from wmh.providers.base import ProviderConfig, ProviderKind, ToolCallingProvider


@dataclass(frozen=True)
class _RunCall:
    instruction: str
    retry_recoverable: bool


class _FakeProject:
    workspace = "/home/user/project"

    def __init__(self, snapshots: list[HarnessSourceTree | Exception]) -> None:
        self.files: dict[str, str] = {}
        self.snapshots = snapshots
        self.staged: list[tuple[HarnessSourceTree, str]] = []
        self.bash_commands: list[str] = []
        self.bash_results: dict[str, ProjectBashResult] = {}
        self.run_calls: list[_RunCall] = []
        self.emit_submit = True
        self.run_raises: Exception | None = None
        self.closed = 0

    def write_text(self, path: str, content: str) -> None:
        self.files[path] = content

    def run_bash(self, command: str) -> ProjectBashResult:
        self.bash_commands.append(command)
        for marker, result in self.bash_results.items():
            if marker in command:
                return result
        return ProjectBashResult(stdout="", stderr="", exit_code=0)

    def stage_source_tree(self, tree: HarnessSourceTree, dest: str) -> None:
        self.staged.append((tree, dest))

    def snapshot_source_tree(
        self, directory: str, *, max_files: int, max_bytes: int
    ) -> HarnessSourceTree:
        del directory, max_files, max_bytes
        item = self.snapshots.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

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
    ) -> AgentProjectRun:
        del agent, provider, should_cancel, writable_files
        self.run_calls.append(_RunCall(instruction, retry_recoverable))
        if self.run_raises is not None:
            raise self.run_raises
        if self.emit_submit and on_event is not None:
            on_event(SessionEvent(kind="submit", payload={"answer": "done"}))
        return AgentProjectRun(answer="done", events=(), worker_usage=TokenUsage())

    def close(self) -> None:
        self.closed += 1


class _Provider:
    config = ProviderConfig(kind=ProviderKind.BEDROCK, model="meta")

    def complete_chat(self, request: object) -> ChatResponse:
        del request
        raise AssertionError("the fake project never calls the provider")


def _tree(prompt: str, *extra: tuple[str, str]) -> HarnessSourceTree:
    files = [HarnessSourceFile(path="SYSTEM.md", content=prompt)]
    files.extend(HarnessSourceFile(path=path, content=content) for path, content in extra)
    return HarnessSourceTree(files=tuple(files))


def _seed_with_trial(tmp_path: Path) -> EvaluatedCandidate:
    trial = tmp_path / "harbor" / "wmh-abc" / "t1__trial"
    (trial / "agent").mkdir(parents=True)
    (trial / "agent" / "wmh-run.json").write_text('{"steps": ["s1"]}', encoding="utf-8")
    (trial / "verifier").mkdir()
    (trial / "verifier" / "output.txt").write_text("PASS", encoding="utf-8")
    (trial / "config.json").write_text("{}", encoding="utf-8")  # ceremony: never copied
    tree = _tree("seed")
    doc = tree.to_doc("candidate-0000")
    report = ScoreReport(
        doc_hash=doc.doc_hash,
        request=ScoreRequest(task_ids=("t1",), attempts=1),
        reward_mode="positive-binary",
        cells=(
            ScoreCell(task_id="t1", attempt=1, reward=1.0, passed=True, artifact_dir=str(trial)),
        ),
    )
    return EvaluatedCandidate("candidate-0000", tree, report)


def _proposer(
    projects: list[_FakeProject],
    run_dir: Path,
    *,
    max_candidate_history_bytes: int = DEFAULT_CANDIDATE_HISTORY_BYTES,
    max_history_file_bytes: int = DEFAULT_HISTORY_FILE_BYTES,
) -> ProjectCandidateProposer:
    return ProjectCandidateProposer(
        optimizer_agent(),
        _Provider(),
        project_factory=lambda: projects.pop(0),
        run_dir=run_dir,
        max_candidate_history_bytes=max_candidate_history_bytes,
        max_history_file_bytes=max_history_file_bytes,
    )


def test_proposer_materializes_history_stages_seed_and_returns_one_candidate(
    tmp_path: Path,
) -> None:
    candidate = _tree("improved", ("src/agent-loop.ts", "export {};"))
    project = _FakeProject([candidate])
    seed = _seed_with_trial(tmp_path)
    run_dir = tmp_path / "run"

    proposal = _proposer([project], run_dir).propose((seed,), slot=1)

    assert proposal.candidate_id == "candidate-0001"
    assert proposal.source == candidate
    assert project.staged == [(seed.source, "candidate")]
    assert project.run_calls[0].retry_recoverable is False
    request = project.run_calls[0].instruction
    assert "/home/user/project/candidate" in request
    assert "kebab-case" in request  # the filename grammar is load-bearing prompt content
    assert "history/candidate-0000/source" in request
    # Complete history: source, report, transcript, and verifier output; NOT harbor ceremony.
    assert project.files["history/candidate-0000/source/SYSTEM.md"] == "seed"
    assert project.files["history/candidate-0000/trials/t1/attempt-1/wmh-run.json"] == (
        '{"steps": ["s1"]}'
    )
    assert project.files["history/candidate-0000/trials/t1/attempt-1/verifier/output.txt"] == (
        "PASS"
    )
    assert not any("config.json" in path for path in project.files)
    report = json.loads(project.files["history/candidate-0000/report.json"])
    assert report["score"] == 1.0
    manifest = json.loads(project.files["history/manifest.json"])
    assert manifest["candidates"][0]["by_task"] == {"t1": 1.0}
    # The interface gate parse-checked the candidate's one code file in ONE node call.
    [check_command] = project.bash_commands
    assert check_command.startswith("node -e ")
    assert "stripTypeScriptTypes" in check_command
    assert check_command.endswith(" candidate/src/agent-loop.ts")
    assert project.closed == 1
    slot_dir = run_dir / "proposals" / "slot-0001"
    assert (slot_dir / "REQUEST.md").is_file()
    assert (slot_dir / "source" / "SYSTEM.md").read_text(encoding="utf-8") == "improved"
    assert json.loads((slot_dir / "status.json").read_text(encoding="utf-8"))["valid"] is True


def test_directive_appears_in_the_request_and_on_event_forwards(tmp_path: Path) -> None:
    candidate = _tree("improved")
    project = _FakeProject([candidate])
    seed = _seed_with_trial(tmp_path)
    run_dir = tmp_path / "run"
    forwarded: list[SessionEvent] = []
    proposer = ProjectCandidateProposer(
        optimizer_agent(),
        _Provider(),
        project_factory=lambda: project,
        run_dir=run_dir,
        directive="the agent should have access to my GitHub",
        on_event=forwarded.append,
    )

    proposer.propose((seed,), slot=1)

    request = project.run_calls[0].instruction
    assert "The user directing this optimization gave the following feedback" in request
    assert "the agent should have access to my GitHub" in request
    # The wrapped sink forwards the same events the project emitted to the CLI callback.
    assert [event.kind for event in forwarded] == ["submit"]


def test_on_event_sink_error_never_aborts_the_proposal(tmp_path: Path) -> None:
    project = _FakeProject([_tree("improved")])
    seed = _seed_with_trial(tmp_path)

    def boom(_event: SessionEvent) -> None:
        raise RuntimeError("sink is broken")

    proposer = ProjectCandidateProposer(
        optimizer_agent(),
        _Provider(),
        project_factory=lambda: project,
        run_dir=tmp_path / "run",
        on_event=boom,
    )

    proposal = proposer.propose((seed,), slot=1)
    assert proposal.candidate_id == "candidate-0001"


def test_directive_validation_rejects_oversized_and_non_durable_text(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exceeds"):
        ProjectCandidateProposer(
            optimizer_agent(),
            _Provider(),
            project_factory=lambda: _FakeProject([]),
            run_dir=tmp_path / "run",
            directive="x" * 20_001,
        )


def test_no_directive_leaves_the_request_free_of_the_feedback_block(tmp_path: Path) -> None:
    project = _FakeProject([_tree("improved")])
    seed = _seed_with_trial(tmp_path)

    _proposer([project], tmp_path / "run").propose((seed,), slot=1)

    request = project.run_calls[0].instruction
    assert "The user directing this optimization" not in request


def test_missing_submit_consumes_the_slot_with_persisted_evidence(tmp_path: Path) -> None:
    project = _FakeProject([_tree("improved")])
    project.emit_submit = False
    seed = _seed_with_trial(tmp_path)
    run_dir = tmp_path / "run"

    with pytest.raises(CandidateProposalError, match="did not submit") as excinfo:
        _proposer([project], run_dir).propose((seed,), slot=1)

    slot_dir = run_dir / "proposals" / "slot-0001"
    assert excinfo.value.evidence_dir == str(slot_dir)
    status = json.loads((slot_dir / "status.json").read_text(encoding="utf-8"))
    assert status["valid"] is False
    assert (slot_dir / "events.json").is_file()


def test_interface_validation_failure_preserves_node_stderr(tmp_path: Path) -> None:
    project = _FakeProject([_tree("improved", ("src/agent-loop.ts", "export {"))])
    project.bash_results["node "] = ProjectBashResult(
        stdout="", stderr="SyntaxError: unexpected end of input", exit_code=1
    )
    seed = _seed_with_trial(tmp_path)

    with pytest.raises(CandidateProposalError, match="SyntaxError") as excinfo:
        _proposer([project], tmp_path / "run").propose((seed,), slot=1)

    assert "interface validation failed" in excinfo.value.reason
    assert "SyntaxError: unexpected end of input" in excinfo.value.reason


def test_agent_transport_failure_consumes_the_slot_instead_of_propagating(
    tmp_path: Path,
) -> None:
    project = _FakeProject([_tree("leftover")])
    project.run_raises = RuntimeError("server disconnected mid-turn")
    seed = _seed_with_trial(tmp_path)

    with pytest.raises(CandidateProposalError, match="server disconnected"):
        _proposer([project], tmp_path / "run").propose((seed,), slot=1)
    assert project.closed == 1


def test_invalid_snapshot_tree_is_slot_evidence(tmp_path: Path) -> None:
    project = _FakeProject([ValueError("paths differ only by letter case")])
    seed = _seed_with_trial(tmp_path)

    with pytest.raises(CandidateProposalError, match="letter case"):
        _proposer([project], tmp_path / "run").propose((seed,), slot=1)


def test_oversized_evidence_is_head_tail_truncated_never_fatal(tmp_path: Path) -> None:
    seed = _seed_with_trial(tmp_path)
    transcript = Path(seed.report.cells[0].artifact_dir) / "agent" / "wmh-run.json"
    transcript.write_text("H" * 200 + "MIDDLE" + "T" * 200, encoding="utf-8")
    project = _FakeProject([_tree("improved")])

    _proposer([project], tmp_path / "run", max_history_file_bytes=64).propose((seed,), slot=1)

    copied = project.files["history/candidate-0000/trials/t1/attempt-1/wmh-run.json"]
    assert "bytes truncated" in copied
    assert copied.startswith("HHH")
    assert copied.endswith("TTT")
    assert "MIDDLE" not in copied


def test_each_slot_gets_a_fresh_project_carrying_prior_proposal_evidence(
    tmp_path: Path,
) -> None:
    seed = _seed_with_trial(tmp_path)
    run_dir = tmp_path / "run"
    first = _FakeProject([_tree("improved")])
    first.emit_submit = False  # slot 1 fails and leaves evidence in the run dir
    second = _FakeProject([_tree("improved")])
    proposer = _proposer([first, second], run_dir)

    with pytest.raises(CandidateProposalError):
        proposer.propose((seed,), slot=1)
    proposal = proposer.propose((seed,), slot=2)

    assert proposal.candidate_id == "candidate-0002"
    assert first.closed == 1 and second.closed == 1
    # The failed turn's trace teaches the next fresh project.
    assert "proposals/slot-0001/status.json" in second.files
    assert "proposals/slot-0001/REQUEST.md" in second.files
    assert not any(path.startswith("proposals/slot-0002") for path in first.files)


_NODE = shutil.which("node")


def _run_ts_check(paths: list[str]) -> subprocess.CompletedProcess[str]:
    assert _NODE is not None
    return subprocess.run(
        [_NODE, "-e", project_proposer._TS_CHECK_SCRIPT, *paths],  # noqa: SLF001
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.skipif(_NODE is None, reason="node is not installed")
def test_interface_validation_script_accepts_every_vendored_pi_source(tmp_path: Path) -> None:
    """Empirical gate check: the seed's own vendored TS/JS must pass, or every slot burns.

    (`node --check` is NOT usable here: it does not strip types under --check on node 22 and
    falsely rejects valid TS such as the vendored pi sources.)
    """
    tree = HarnessSourceTree.from_doc(default_agent())
    code_paths: list[str] = []
    for item in tree.files:
        if not item.path.endswith((".ts", ".js")):
            continue
        target = tmp_path / item.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(item.content, encoding="utf-8")
        code_paths.append(str(target))
    assert len(code_paths) >= 20  # the whole vendored agent, not a sample

    done = _run_ts_check(code_paths)

    if project_proposer._TS_VALIDATION_SKIP_MARKER in done.stdout:  # noqa: SLF001
        pytest.skip("local node lacks node:module.stripTypeScriptTypes")
    assert done.returncode == 0, done.stderr


@pytest.mark.skipif(_NODE is None, reason="node is not installed")
def test_interface_validation_script_rejects_broken_typescript_with_the_node_error(
    tmp_path: Path,
) -> None:
    good = tmp_path / "good.ts"
    good.write_text("export function fine(value: string): string { return value; }\n")
    bad = tmp_path / "bad.ts"
    bad.write_text("export function broken(: string {\n")

    done = _run_ts_check([str(good), str(bad)])

    if project_proposer._TS_VALIDATION_SKIP_MARKER in done.stdout:  # noqa: SLF001
        pytest.skip("local node lacks node:module.stripTypeScriptTypes")
    assert done.returncode == 1
    assert "bad.ts" in done.stderr  # per-file attribution with node's own error preserved
    assert "good.ts" not in done.stderr
    assert "Expected" in done.stderr or "Unexpected" in done.stderr


def test_evidence_gaps_are_marked_per_trial_and_for_prior_proposals(tmp_path: Path) -> None:
    """Truncation honesty: missing, truncated, and omitted evidence is marked where it lives."""
    seed = _seed_with_trial(tmp_path)
    no_evidence_cell = ScoreCell(task_id="t2", attempt=1, reward=0.0, passed=False)
    tree = seed.source
    doc = tree.to_doc("candidate-0000")
    report = ScoreReport(
        doc_hash=doc.doc_hash,
        request=ScoreRequest(task_ids=("t1", "t2"), attempts=1),
        reward_mode="positive-binary",
        cells=(seed.report.cells[0], no_evidence_cell),
    )
    seed = EvaluatedCandidate("candidate-0000", tree, report)
    transcript = Path(seed.report.cells[0].artifact_dir) / "agent" / "wmh-run.json"
    transcript.write_text("H" * 200 + "T" * 200, encoding="utf-8")
    run_dir = tmp_path / "run"
    first = _FakeProject([_tree("improved")])
    first.emit_submit = False
    proposer = _proposer([first], run_dir, max_history_file_bytes=64)
    with pytest.raises(CandidateProposalError):
        proposer.propose((seed,), slot=1)

    truncated_marker = first.files[
        "history/candidate-0000/trials/t1/attempt-1/EVIDENCE-TRUNCATED.md"
    ]
    assert "head/tail truncated" in truncated_marker
    assert "wmh-run.json" in truncated_marker
    assert (
        "no raw evidence"
        in (first.files["history/candidate-0000/trials/t2/attempt-1/NO-EVIDENCE.md"])
    )

    # A second slot with an exhausted prior-proposal budget marks the omissions too.
    second = _FakeProject([_tree("improved")])
    tight = _proposer([second], run_dir, max_candidate_history_bytes=1)
    tight.propose((seed,), slot=2)
    assert "slot-0001" in second.files["proposals/TRUNCATED.md"]


def test_crashed_slot_evidence_is_set_aside_not_deleted(tmp_path: Path) -> None:
    """A redone slot keeps the crashed attempt's raw evidence under attempt-K/."""
    seed = _seed_with_trial(tmp_path)
    run_dir = tmp_path / "run"
    slot_dir = run_dir / "proposals" / "slot-0001"
    slot_dir.mkdir(parents=True)
    (slot_dir / "REQUEST.md").write_text("crashed attempt request", encoding="utf-8")
    project = _FakeProject([_tree("improved")])

    proposal = _proposer([project], run_dir).propose((seed,), slot=1)

    assert proposal.candidate_id == "candidate-0001"
    aside = slot_dir / "attempt-1" / "REQUEST.md"
    assert aside.read_text(encoding="utf-8") == "crashed attempt request"
    assert (slot_dir / "REQUEST.md").read_text(encoding="utf-8") != "crashed attempt request"
