"""CLI tests for feedback-directed harness improvement (execution seam monkeypatched)."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import cast

import pytest
from typer.testing import CliRunner

from wmh.cli import app
from wmh.cli.harness_improve import HarnessImproveOutcome, ImproveAlreadySatisfied
from wmh.config.settings import ModelRole, ModelsSettings, ProjectSettings, save_settings
from wmh.harness.improve import ImproveGate, ImproveOutcome, VerificationResult
from wmh.harness.population import EvaluatedCandidate, PopulationResult, SlotOutcome
from wmh.harness.scoring import ScoreCell, ScoreReport, ScoreRequest
from wmh.harness.source_tree import HarnessSourceFile, HarnessSourceTree
from wmh.providers.base import ProviderConfig

module = importlib.import_module("wmh.cli.harness_improve")
runner = CliRunner()


def _source(prompt: str) -> HarnessSourceTree:
    return HarnessSourceTree(
        files=(
            HarnessSourceFile(path="SYSTEM.md", content=prompt),
            HarnessSourceFile(
                path="config.toml",
                content='[harness]\ntools = ["bash", "submit"]\nruntime_kind = "pi-node"\n',
            ),
        )
    )


def _evaluated(candidate_id: str, source: HarnessSourceTree) -> EvaluatedCandidate:
    doc = source.to_doc(candidate_id)
    request = ScoreRequest(task_ids=("suite-a", "feedback-verify-01"), attempts=1)
    report = ScoreReport(
        doc_hash=doc.doc_hash,
        request=request,
        reward_mode="positive-binary",
        cells=(
            ScoreCell(task_id="suite-a", attempt=1, reward=0.8, passed=True, note="suite"),
            ScoreCell(
                task_id="feedback-verify-01", attempt=1, reward=1.0, passed=True, note="verify"
            ),
        ),
    )
    return EvaluatedCandidate(candidate_id=candidate_id, source=source, report=report)


def _accepted_outcome() -> ImproveOutcome:
    evaluated = _evaluated("candidate-0000", _source("seed"))
    result = PopulationResult(
        outcomes=(SlotOutcome(slot=0, candidate_id="candidate-0000", evaluated=evaluated),),
        population=(evaluated,),
        best=evaluated,
        completed=True,
    )
    return ImproveOutcome(
        accepted=True,
        reason="candidate-0000 passed the gate",
        seed_suite_score=0.7,
        candidate_suite_score=0.8,
        verification=(
            VerificationResult(task_id="feedback-verify-01", passed=True, pass_count=1, attempts=1),
        ),
        selected=evaluated,
        result=result,
    )


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / ".wmh"
    save_settings(
        ProjectSettings(
            models=ModelsSettings(
                meta=ModelRole(provider="openai_responses", model="meta-model"),
            )
        ),
        root,
    )
    tasks_path = tmp_path / "suite.jsonl"
    tasks_path.write_text(
        '{"task_id": "suite-a", "instruction": "do a", "gold": ["done"]}\n',
        encoding="utf-8",
    )
    return root, tasks_path


def test_cli_requires_exactly_one_feedback_source(tmp_path: Path) -> None:
    _root, tasks_path = _write_inputs(tmp_path)
    run_dir = tmp_path / "run"

    neither = runner.invoke(
        app,
        [
            "harness",
            "improve",
            "helper",
            "--tasks",
            str(tasks_path),
            "--model",
            "wm",
            "--run-dir",
            str(run_dir),
        ],
    )
    assert neither.exit_code == 2
    assert "exactly one of --feedback or --feedback-file" in neither.output

    feedback_file = tmp_path / "feedback.txt"
    feedback_file.write_text("give me GitHub access\n", encoding="utf-8")
    both = runner.invoke(
        app,
        [
            "harness",
            "improve",
            "helper",
            "--tasks",
            str(tasks_path),
            "--model",
            "wm",
            "--run-dir",
            str(run_dir),
            "--feedback",
            "text",
            "--feedback-file",
            str(feedback_file),
        ],
    )
    assert both.exit_code == 2
    assert "exactly one of --feedback or --feedback-file" in both.output


def test_cli_requires_explicit_confirmation_in_noninteractive_mode(tmp_path: Path) -> None:
    root, tasks_path = _write_inputs(tmp_path)

    invoked = runner.invoke(
        app,
        [
            "harness",
            "improve",
            "helper",
            "--feedback",
            "give me GitHub access",
            "--tasks",
            str(tasks_path),
            "--model",
            "wm",
            "--run-dir",
            str(tmp_path / "run"),
            "--root",
            str(root),
        ],
    )

    assert invoked.exit_code == 2
    assert "pass --yes" in invoked.output


def test_cli_composes_inputs_and_prints_the_accepted_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, tasks_path = _write_inputs(tmp_path)
    feedback_file = tmp_path / "feedback.txt"
    feedback_file.write_text("give me GitHub access\n", encoding="utf-8")
    run_dir = tmp_path / "improve-run"
    calls: list[dict[str, object]] = []
    outcome = _accepted_outcome()

    def fake_execute(**kwargs: object) -> HarnessImproveOutcome:
        calls.append(kwargs)
        assert outcome.selected is not None
        saved = outcome.selected.candidate.model_copy(update={"name": "helper", "version": 4})
        return HarnessImproveOutcome(
            outcome=outcome,
            saved=saved,
            run_dir=cast("Path", kwargs["run_dir"]),
            dropped=("feedback-verify-02",),
        )

    monkeypatch.setattr(module, "_execute_improvement", fake_execute)

    invoked = runner.invoke(
        app,
        [
            "harness",
            "improve",
            "helper",
            "--feedback-file",
            str(feedback_file),
            "--tasks",
            str(tasks_path),
            "--model",
            "wm",
            "--verification-count",
            "2",
            "--margin",
            "0.2",
            "--iterations",
            "3",
            "--attempts",
            "5",
            "--seed",
            "other@2",
            "--no-seed-knowledge",
            "--run-dir",
            str(run_dir),
            "--root",
            str(root),
            "--yes",
        ],
    )

    assert invoked.exit_code == 0, invoked.output
    normalized = " ".join(invoked.output.split())
    assert "improved" in normalized
    assert "v4" in normalized
    assert "verification 1/1 passed" in normalized
    [call] = calls
    assert call["name"] == "helper"
    assert call["feedback"] == "give me GitHub access\n"
    assert call["model"] == "wm"
    assert call["verification_count"] == 2
    assert call["gate"] == ImproveGate(suite_margin=0.2)
    assert call["iterations"] == 3
    assert call["attempts"] == 5
    assert call["seed_reference"] == "other@2"
    assert call["seed_knowledge"] is False
    assert call["run_dir"] == run_dir
    assert cast("ProviderConfig", call["meta_config"]).model == "meta-model"


def test_cli_reports_rejection_and_already_satisfied_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, tasks_path = _write_inputs(tmp_path)
    rejected = ImproveOutcome(
        accepted=False,
        reason="suite regression beyond margin: best candidate candidate-0001 scored 0.5",
        seed_suite_score=0.8,
        candidate_suite_score=None,
        verification=(),
        selected=None,
        result=_accepted_outcome().result,
    )

    def fake_rejected(**kwargs: object) -> HarnessImproveOutcome:
        return HarnessImproveOutcome(
            outcome=rejected,
            saved=None,
            run_dir=cast("Path", kwargs["run_dir"]),
            dropped=(),
        )

    monkeypatch.setattr(module, "_execute_improvement", fake_rejected)
    argv = [
        "harness",
        "improve",
        "helper",
        "--feedback",
        "give me GitHub access",
        "--tasks",
        str(tasks_path),
        "--model",
        "wm",
        "--run-dir",
        str(tmp_path / "run"),
        "--root",
        str(root),
        "--yes",
    ]

    invoked = runner.invoke(app, argv)
    assert invoked.exit_code == 0, invoked.output
    assert "rejected" in invoked.output
    assert "suite regression beyond margin" in invoked.output

    def fake_satisfied(**kwargs: object) -> ImproveAlreadySatisfied:
        return ImproveAlreadySatisfied(
            dropped=("feedback-verify-01",),
            run_dir=cast("Path", kwargs["run_dir"]),
        )

    monkeypatch.setattr(module, "_execute_improvement", fake_satisfied)
    satisfied = runner.invoke(app, argv)
    assert satisfied.exit_code == 1
    assert "already satisfied" in satisfied.output
    assert "feedback-verify-01" in satisfied.output


def test_cli_rejects_invalid_margin_and_existing_run_dir(tmp_path: Path) -> None:
    root, tasks_path = _write_inputs(tmp_path)
    base = [
        "harness",
        "improve",
        "helper",
        "--feedback",
        "give me GitHub access",
        "--tasks",
        str(tasks_path),
        "--model",
        "wm",
        "--root",
        str(root),
        "--yes",
    ]

    bad_margin = runner.invoke(app, [*base, "--run-dir", str(tmp_path / "m"), "--margin", "1.0"])
    assert bad_margin.exit_code == 2
    assert "--margin" in bad_margin.output

    taken = tmp_path / "taken"
    taken.mkdir()
    (taken / "state.json").write_text("{}", encoding="utf-8")
    existing = runner.invoke(app, [*base, "--run-dir", str(taken)])
    assert existing.exit_code == 2
    # The panel wraps and inserts box borders, so assert on a short unbroken tail of the message.
    assert "run-dir" in existing.output


def test_cli_help_names_the_gate_and_feedback_inputs() -> None:
    invoked = runner.invoke(app, ["harness", "improve", "--help"])

    assert invoked.exit_code == 0, invoked.output
    # The 80-column help panel truncates long option names, so assert stable prefixes.
    assert "--feedback" in invoked.output
    assert "--verification" in invoked.output
    assert "--margin" in invoked.output
    assert "--seed-knowled" in invoked.output
    assert "--run-dir" in invoked.output
    assert "models.meta" in invoked.output
    assert "name@ref" in invoked.output
