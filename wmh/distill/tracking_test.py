"""Tests for the wandb tracking seam, against a recording fake wandb module.

The fake is installed into `sys.modules["wandb"]` so `WandbTracker`'s lazy
import resolves to it; no test talks to the real SDK or network. Credential
resolution is tested against a temporary HOME so the developer's real
~/.netrc can never leak into (or satisfy) an assertion.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Literal

import pytest

from wmh.core.types import JsonObject, JsonValue
from wmh.distill.config import (
    DistillConfig,
    HarborConfig,
    StudentConfig,
    TeacherConfig,
    WandbConfig,
)
from wmh.distill.loop import StepMetrics, WarmupMetrics
from wmh.distill.offpolicy import OffPolicyMetrics
from wmh.distill.samples import SampleRollout
from wmh.distill.tracking import (
    WANDB_API_KEY_ENV,
    WANDB_RUN_FILE,
    NullTracker,
    WandbTracker,
    build_tracker,
)

_AGENT = "pi"


class _FakeSummary:
    """Records `summary.update` payloads."""

    def __init__(self) -> None:
        self.updates: list[dict[str, JsonValue]] = []

    def update(self, values: Mapping[str, JsonValue]) -> None:
        self.updates.append(dict(values))


class _FakeRun:
    """The object the fake `wandb.init` returns."""

    def __init__(self) -> None:
        self.id = "fake-run-0"
        self.summary = _FakeSummary()


class _FakeTable:
    """Records the columns and rows one fake `wandb.Table` was built with."""

    def __init__(self, columns: list[str], data: list[list[JsonValue]]) -> None:
        self.columns = columns
        self.data = data


class _FakeWandb(ModuleType):
    """A recording stand-in for the wandb module (installed in sys.modules)."""

    def __init__(self) -> None:
        super().__init__("wandb")
        self.init_calls: list[JsonObject] = []
        self.log_calls: list[tuple[dict[str, JsonValue | _FakeTable], int]] = []
        self.finish_calls = 0
        self.run = _FakeRun()
        self.fail_logging = False

    def init(
        self,
        *,
        project: str,
        entity: str | None,
        name: str,
        tags: list[str],
        dir: str,  # the wandb SDK's own keyword names, `id` included
        config: JsonObject,
        id: str | None = None,
        resume: str | None = None,
    ) -> _FakeRun:
        self.init_calls.append(
            {
                "project": project,
                "entity": entity,
                "name": name,
                "tags": list(tags),
                "dir": dir,
                "config": config,
                "id": id,
                "resume": resume,
            }
        )
        # Like the real SDK under resume="allow": an explicit id becomes the
        # run's id; otherwise a fresh id is assigned.
        self.run.id = id if id is not None else f"fake-run-{len(self.init_calls)}"
        return self.run

    def Table(self, *, columns: list[str], data: list[list[JsonValue]]) -> _FakeTable:
        if self.fail_logging:
            raise RuntimeError("wandb service unavailable")
        return _FakeTable(columns, data)

    def log(self, data: Mapping[str, JsonValue | _FakeTable], *, step: int) -> None:
        if self.fail_logging:
            raise RuntimeError("wandb service unavailable")
        self.log_calls.append((dict(data), step))

    def finish(self) -> None:
        if self.fail_logging:
            raise RuntimeError("wandb service unavailable")
        self.finish_calls += 1


def _cfg(wandb: WandbConfig | None = None) -> DistillConfig:
    return DistillConfig(
        student=StudentConfig(base_model="fake/student-4b"),
        teacher=TeacherConfig(model="fake/teacher-70b"),
        harbor=HarborConfig(job_template="jobs/tb2.yaml"),
        wandb=wandb if wandb is not None else WandbConfig(enabled=True),
    )


def _metrics(
    *,
    reverse_kl: float | None = -0.25,
    pg_loss: float | None = 1.5,
    grad_norm: float | None = 2.25,
    loss: Literal["importance_sampling", "ppo", "topk_ce"] = "ppo",
    entropy: float | None = 0.181,
    generation_tokens: float | None = 7577.0,
) -> StepMetrics:
    return StepMetrics(
        loss=loss,
        tasks=2,
        trials=4,
        solve_rate=0.5,
        graded_solve_rate=0.625,
        graded_trials=2,
        raw_solve_rate=0.25,
        executed_trials=2,
        infra_failed_trials=2,
        scaffold_loss_rate=0.5,
        stop_reason_counts={"no_tool_call": 1, "submitted": 1, "unknown": 2},
        empty_span_trials=0,
        datums=4,
        fragments=0,
        fragmentation_rate=0.0,
        overflow_drops=0,
        overlong_drops=1,
        mismatch_drops=0,
        clipped_tokens=3,
        loss_tokens=40,
        context_tokens=200,
        reverse_kl_per_token=reverse_kl,
        entropy_per_token=entropy,
        mean_generation_tokens=generation_tokens,
        entropy_baseline=0.2,
        entropy_ratio=None if entropy is None else entropy / 0.2,
        generation_tokens_baseline=8000.0,
        generation_tokens_ratio=None if generation_tokens is None else generation_tokens / 8000.0,
        reward_mean=0.5,
        advantage_mean=0.05,
        advantage_std=1.2,
        clip_fraction=0.075,
        pg_loss=pg_loss,
        grad_norm=grad_norm,
        sampler_path="tinker://fake/sampler/0001",
        student_prefill_tokens=120,
        student_cached_prefill_tokens=30,
        student_sample_tokens=40,
        student_train_tokens=200,
        teacher_prefill_tokens=160,
        teacher_cached_prefill_tokens=0,
        teacher_sample_tokens=0,
        usd=0.75,
        cumulative_usd=3.25,
    )


def _sample(trial_name: str = "task-a__s1", reward: float = 1.0) -> SampleRollout:
    return SampleRollout(
        trial_name=trial_name,
        reward=reward,
        text=f"### trial {trial_name}\n<|im_start|>user\nhi<|im_end|>\n",
    )


def _warmup_metrics() -> WarmupMetrics:
    return WarmupMetrics(
        tasks=4,
        trials=8,
        kept_trials=3,
        solve_rate=0.375,
        datums=3,
        loss_tokens=30,
        context_tokens=90,
        learning_rate=1e-4,
        student_prefill_tokens=0,
        student_cached_prefill_tokens=0,
        student_sample_tokens=0,
        student_train_tokens=120,
        teacher_prefill_tokens=400,
        teacher_cached_prefill_tokens=90,
        teacher_sample_tokens=80,
        usd=0.5,
    )


def _offpolicy_metrics(step: int = 0) -> OffPolicyMetrics:
    return OffPolicyMetrics(
        epoch=step,
        epochs=2,
        minibatch=0,
        planned_steps=2,
        tasks=4,
        trials=8,
        kept_trials=3,
        solve_rate=0.375,
        corpus_datums=3,
        datums=3,
        loss_tokens=30,
        context_tokens=90,
        learning_rate=1e-4,
        loss=1.25,
        grad_norm=None,
        student_prefill_tokens=0,
        student_cached_prefill_tokens=0,
        student_sample_tokens=0,
        student_train_tokens=120,
        teacher_prefill_tokens=400,
        teacher_cached_prefill_tokens=90,
        teacher_sample_tokens=80,
        usd=0.5,
    )


@pytest.fixture
def fake_wandb(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _FakeWandb:
    """A recording wandb module with credentials present, in a clean HOME."""
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv(WANDB_API_KEY_ENV, "test-key")
    return fake


def _tracker(fake: _FakeWandb, run_dir: Path, cfg: DistillConfig | None = None) -> WandbTracker:
    del fake  # already installed by the fixture; kept for call-site clarity
    return WandbTracker(cfg if cfg is not None else _cfg(), run_dir, _AGENT)


# -- build_tracker ---------------------------------------------------------------------------


def test_build_tracker_disabled_is_a_null_tracker(tmp_path: Path) -> None:
    tracker = build_tracker(_cfg(WandbConfig()), tmp_path / "run", _AGENT)
    assert isinstance(tracker, NullTracker)
    # Every NullTracker call is a harmless no-op.
    tracker.log_step(0, _metrics())
    tracker.log_offpolicy_step(0, _offpolicy_metrics())
    tracker.log_warmup_step(0, _warmup_metrics())
    tracker.log_eval("baseline-teacher", 0.5, None)
    tracker.log_samples("train", 0, [_sample()])
    tracker.log_summary(
        gate_accepted=True,
        gate_reason="ok",
        teacher_solve_rate=0.5,
        student_before_solve_rate=0.25,
        student_after_solve_rate=0.5,
        total_usd=1.0,
        steps_completed=3,
    )
    tracker.finish()


def test_build_tracker_enabled_builds_a_wandb_tracker(
    fake_wandb: _FakeWandb, tmp_path: Path
) -> None:
    tracker = build_tracker(_cfg(), tmp_path / "run", _AGENT)
    assert isinstance(tracker, WandbTracker)
    assert len(fake_wandb.init_calls) == 1


# -- init ------------------------------------------------------------------------------------


def test_init_kwargs_carry_the_config_section(fake_wandb: _FakeWandb, tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "smoke-01"
    cfg = _cfg(
        WandbConfig(
            enabled=True,
            project="my-project",
            entity="my-team",
            run_name="explicit-name",
            tags=["smoke", "tb2"],
        )
    )
    _tracker(fake_wandb, run_dir, cfg)
    (call,) = fake_wandb.init_calls
    assert call["project"] == "my-project"
    assert call["entity"] == "my-team"
    assert call["name"] == "explicit-name"
    assert call["tags"] == ["smoke", "tb2"]
    assert call["dir"] == str(run_dir)
    assert run_dir.is_dir()  # wandb needs its dir to exist
    # The run config is the same plain dict snapshot_toml renders.
    assert call["config"] == cfg.model_dump(mode="json", exclude_none=True)


def test_default_run_name_derives_from_agent_and_run_dir(
    fake_wandb: _FakeWandb, tmp_path: Path
) -> None:
    _tracker(fake_wandb, tmp_path / "runs" / "smoke-02")
    (call,) = fake_wandb.init_calls
    assert call["name"] == "pi-smoke-02"


# -- resume across restarts --------------------------------------------------------------------


def test_first_init_starts_fresh_and_persists_the_run_id(
    fake_wandb: _FakeWandb, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    _tracker(fake_wandb, run_dir)
    (call,) = fake_wandb.init_calls
    assert call["id"] is None
    assert call["resume"] is None
    record = json.loads((run_dir / WANDB_RUN_FILE).read_text(encoding="utf-8"))
    assert record == {"run_id": fake_wandb.run.id}


def test_restart_resumes_the_persisted_run_id(fake_wandb: _FakeWandb, tmp_path: Path) -> None:
    """A restarted run CONTINUES the same dashboard run (task #11)."""
    run_dir = tmp_path / "run"
    _tracker(fake_wandb, run_dir)
    first_id = fake_wandb.run.id
    _tracker(fake_wandb, run_dir)  # the restarted session
    second = fake_wandb.init_calls[1]
    assert second["id"] == first_id
    assert second["resume"] == "allow"
    # The record still names the same run after the resume.
    record = json.loads((run_dir / WANDB_RUN_FILE).read_text(encoding="utf-8"))
    assert record == {"run_id": first_id}


def test_corrupt_run_record_falls_back_to_a_fresh_run_and_rewrites(
    fake_wandb: _FakeWandb, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / WANDB_RUN_FILE).write_text("{not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="wmh.distill.tracking"):
        _tracker(fake_wandb, run_dir)
    assert "corrupt wandb run record" in caplog.text
    (call,) = fake_wandb.init_calls
    assert call["id"] is None
    assert call["resume"] is None
    # The broken file was replaced with the fresh run's id.
    record = json.loads((run_dir / WANDB_RUN_FILE).read_text(encoding="utf-8"))
    assert record == {"run_id": fake_wandb.run.id}


def test_a_torn_run_record_write_leaves_the_previous_record_intact(
    fake_wandb: _FakeWandb, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The persisted run id is the resume anchor, so its write must be atomic.

    A crash mid-write is modelled by a `write_text` that persists half its
    payload and then raises. The record goes through the run dir's shared
    temp-file-then-replace helper, so the half write lands on the temp file
    and the record a resume reads back is still the complete previous one; an
    in-place write would have truncated it and cost the dashboard run.
    """
    run_dir = tmp_path / "run"
    _tracker(fake_wandb, run_dir)
    first_id = fake_wandb.run.id
    intact = Path.write_text

    def _torn(
        self: Path,
        data: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        intact(self, data[: len(data) // 2], encoding=encoding)
        raise OSError("disk full halfway through the record")

    monkeypatch.setattr(Path, "write_text", _torn)
    with pytest.raises(OSError, match="disk full"):
        _tracker(fake_wandb, run_dir)

    assert json.loads((run_dir / WANDB_RUN_FILE).read_text(encoding="utf-8")) == {
        "run_id": first_id
    }
    # The torn bytes went to the sibling temp file, never to the record.
    assert (run_dir / f"{WANDB_RUN_FILE}.tmp").is_file()


def test_missing_sdk_error_names_the_extra(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # None in sys.modules makes `import wandb` raise ImportError (the halted-import
    # convention), simulating an environment without the distill extra.
    monkeypatch.setitem(sys.modules, "wandb", None)
    monkeypatch.setenv(WANDB_API_KEY_ENV, "test-key")
    with pytest.raises(ImportError, match=r"uv sync --extra distill"):
        WandbTracker(_cfg(), tmp_path / "run", _AGENT)


# -- credentials -----------------------------------------------------------------------------


def test_missing_credentials_error_names_both_fixes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "wandb", _FakeWandb())
    monkeypatch.delenv(WANDB_API_KEY_ENV, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))  # no ~/.netrc at all
    with pytest.raises(ValueError, match=r"WANDB_API_KEY.*wandb login") as excinfo:
        WandbTracker(_cfg(), tmp_path / "run", _AGENT)
    message = str(excinfo.value)
    assert "api.wandb.ai" in message
    assert ".netrc" in message


def test_netrc_without_a_wandb_machine_is_not_a_credential(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "wandb", _FakeWandb())
    monkeypatch.delenv(WANDB_API_KEY_ENV, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    netrc_path = home / ".netrc"
    netrc_path.write_text("machine example.com login user password hunter2\n", encoding="utf-8")
    os.chmod(netrc_path, 0o600)  # netrc rejects world-readable password files
    monkeypatch.setenv("HOME", str(home))
    with pytest.raises(ValueError, match="no credentials were found"):
        WandbTracker(_cfg(), tmp_path / "run", _AGENT)


def test_netrc_wandb_login_satisfies_the_credential_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.delenv(WANDB_API_KEY_ENV, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    netrc_path = home / ".netrc"
    netrc_path.write_text("machine api.wandb.ai login user password test-key\n", encoding="utf-8")
    os.chmod(netrc_path, 0o600)
    monkeypatch.setenv("HOME", str(home))
    WandbTracker(_cfg(), tmp_path / "run", _AGENT)
    assert len(fake.init_calls) == 1


# -- logging ---------------------------------------------------------------------------------


def test_log_step_flattens_the_metrics_row(fake_wandb: _FakeWandb, tmp_path: Path) -> None:
    tracker = _tracker(fake_wandb, tmp_path / "run")
    tracker.log_step(3, _metrics())
    (payload, step) = fake_wandb.log_calls[-1]
    assert step == 3
    assert payload == {
        "train/tasks": 2,
        "train/trials": 4,
        "train/solve_rate": 0.5,
        # The graded companion charts beside the binary rate, with its own denominator: a graded
        # rate over zero graded trials is a null measurement, not a 0.0 datapoint.
        "train/graded_solve_rate": 0.625,
        "train/graded_trials": 2,
        "train/raw_solve_rate": 0.25,
        "train/executed_trials": 2,
        "train/infra_failed_trials": 2,
        # The audit headline gets its own chartable key, plus one series per stop reason, so a
        # harness-induced floor can never sit at 88.8% unnoticed again.
        "train/scaffold_loss_rate": 0.5,
        "stop/no_tool_call": 1,
        "stop/submitted": 1,
        "stop/unknown": 2,
        "train/empty_span_trials": 0,
        "train/truncated_spans": 0,
        "train/datums": 4,
        "train/fragments": 0,
        "train/fragmentation_rate": 0.0,
        "train/overflow_drops": 0,
        "train/overlong_drops": 1,
        "train/mismatch_drops": 0,
        "train/clipped_tokens": 3,
        "train/loss_tokens": 40,
        "train/context_tokens": 200,
        "train/reverse_kl_per_token": -0.25,
        # The degeneration pair, each with the baseline it is judged against and
        # the ratio the tripwire actually bounds (the cross-run comparable one).
        "train/entropy_per_token": 0.181,
        "train/mean_generation_tokens": 7577.0,
        "train/entropy_baseline": 0.2,
        "train/entropy_ratio": 0.181 / 0.2,
        "train/generation_tokens_baseline": 8000.0,
        "train/generation_tokens_ratio": 7577.0 / 8000.0,
        "train/reward_mean": 0.5,
        # The one string key kept: advantage_mean and clip_fraction mean
        # different things per objective, so the row says which one ran.
        "train/loss": "ppo",
        "train/advantage_mean": 0.05,
        "train/advantage_std": 1.2,
        "train/clip_fraction": 0.075,
        "train/pg_loss": 1.5,
        "train/grad_norm": 2.25,
        "tokens/student_prefill": 120,
        "tokens/student_cached_prefill": 30,
        "tokens/student_sample": 40,
        "tokens/student_train": 200,
        "tokens/teacher_prefill": 160,
        "tokens/teacher_cached_prefill": 0,
        "tokens/teacher_sample": 0,
        "cost/usd": 0.75,
        "cost/usd_cum": 3.25,
    }
    # The sampler path (a string) never lands in a chartable payload.
    assert not any("sampler" in key for key in payload)


@pytest.mark.parametrize("loss", ["importance_sampling", "ppo", "topk_ce"])
def test_log_step_distinguishes_every_objective(
    fake_wandb: _FakeWandb, tmp_path: Path, loss: Literal["importance_sampling", "ppo", "topk_ce"]
) -> None:
    """Each of the three modes is identifiable from its own dashboard row.

    Without this, `train/advantage_mean` and `train/clip_fraction` would be
    unreadable across runs: the same key means the centered near-zero baseline
    under one objective, the raw teacher-minus-student gap under another, and
    nothing at all under topk_ce.
    """
    tracker = _tracker(fake_wandb, tmp_path / "run")
    tracker.log_step(0, _metrics(loss=loss))
    (payload, _) = fake_wandb.log_calls[-1]
    assert payload["train/loss"] == loss


def test_log_step_carries_both_cost_keys(fake_wandb: _FakeWandb, tmp_path: Path) -> None:
    """The per-row delta AND the all-session cumulative total chart together."""
    tracker = _tracker(fake_wandb, tmp_path / "run")
    tracker.log_step(0, _metrics())
    (payload, _) = fake_wandb.log_calls[-1]
    assert payload["cost/usd"] == 0.75
    assert payload["cost/usd_cum"] == 3.25


def test_log_step_drops_an_unscored_reverse_kl(fake_wandb: _FakeWandb, tmp_path: Path) -> None:
    tracker = _tracker(fake_wandb, tmp_path / "run")
    tracker.log_step(0, _metrics(reverse_kl=None))
    (payload, _) = fake_wandb.log_calls[-1]
    assert "train/reverse_kl_per_token" not in payload


def test_log_step_drops_unmeasured_degeneration_metrics(
    fake_wandb: _FakeWandb, tmp_path: Path
) -> None:
    """A batch that sampled nothing charts no entropy or length point: a zero
    there would read as total collapse on the dashboard."""
    tracker = _tracker(fake_wandb, tmp_path / "run")
    tracker.log_step(0, _metrics(entropy=None, generation_tokens=None))
    (payload, _) = fake_wandb.log_calls[-1]
    assert "train/entropy_per_token" not in payload
    assert "train/mean_generation_tokens" not in payload
    assert "train/entropy_ratio" not in payload
    assert "train/generation_tokens_ratio" not in payload
    # The baselines still chart: they are the run's fixed reference, not a
    # measurement of this step.
    assert payload["train/entropy_baseline"] == 0.2
    assert payload["train/generation_tokens_baseline"] == 8000.0


def test_log_step_omits_backend_metrics_the_sdk_never_reported(
    fake_wandb: _FakeWandb, tmp_path: Path
) -> None:
    """None-valued pg_loss/grad_norm simply do not chart (never fabricated)."""
    tracker = _tracker(fake_wandb, tmp_path / "run")
    tracker.log_step(0, _metrics(pg_loss=None, grad_norm=None))
    (payload, _) = fake_wandb.log_calls[-1]
    assert "train/pg_loss" not in payload
    assert "train/grad_norm" not in payload
    # The computable metrics still chart.
    assert payload["train/clip_fraction"] == 0.075
    assert payload["cost/usd_cum"] == 3.25


def test_log_warmup_step_uses_warmup_keys_at_wandb_step_zero(
    fake_wandb: _FakeWandb, tmp_path: Path
) -> None:
    """Warmup rows chart under warmup/* and always at wandb step 0.

    Warmup precedes training step 0 and wandb steps must never decrease, so
    logging warmup rows at their own indices would make wandb drop the later
    train/ rows; the warmup index rides inside the payload instead.
    """
    tracker = _tracker(fake_wandb, tmp_path / "run")
    tracker.log_warmup_step(0, _warmup_metrics())
    tracker.log_warmup_step(1, _warmup_metrics())
    assert [step for _, step in fake_wandb.log_calls] == [0, 0]
    (payload, _) = fake_wandb.log_calls[-1]
    assert payload == {
        "warmup/step": 1,
        "warmup/tasks": 4,
        "warmup/trials": 8,
        "warmup/kept_trials": 3,
        "warmup/solve_rate": 0.375,
        "warmup/datums": 3,
        "warmup/loss_tokens": 30,
        "warmup/context_tokens": 90,
        "warmup/learning_rate": 1e-4,
        "warmup/student_prefill_tokens": 0,
        "warmup/student_cached_prefill_tokens": 0,
        "warmup/student_sample_tokens": 0,
        "warmup/student_train_tokens": 120,
        "warmup/teacher_prefill_tokens": 400,
        "warmup/teacher_cached_prefill_tokens": 90,
        "warmup/teacher_sample_tokens": 80,
        "warmup/usd": 0.5,
    }
    # The constant phase discriminator (a string) never lands in the payload.
    assert not any("phase" in key for key in payload)


def test_log_offpolicy_step_uses_offpolicy_keys_at_wandb_step_zero(
    fake_wandb: _FakeWandb, tmp_path: Path
) -> None:
    """Off-policy rows follow the same pre-training rule as the warmup rows."""
    tracker = _tracker(fake_wandb, tmp_path / "run")
    tracker.log_offpolicy_step(0, _offpolicy_metrics(0))
    tracker.log_offpolicy_step(1, _offpolicy_metrics(1))
    assert [step for _, step in fake_wandb.log_calls] == [0, 0]
    (payload, _) = fake_wandb.log_calls[-1]
    assert payload == {
        "offpolicy/step": 1,
        "offpolicy/epoch": 1,
        "offpolicy/epochs": 2,
        "offpolicy/minibatch": 0,
        "offpolicy/planned_steps": 2,
        "offpolicy/tasks": 4,
        "offpolicy/trials": 8,
        "offpolicy/kept_trials": 3,
        "offpolicy/solve_rate": 0.375,
        "offpolicy/corpus_datums": 3,
        "offpolicy/datums": 3,
        "offpolicy/loss_tokens": 30,
        "offpolicy/context_tokens": 90,
        "offpolicy/learning_rate": 1e-4,
        "offpolicy/loss": 1.25,
        "offpolicy/student_prefill_tokens": 0,
        "offpolicy/student_cached_prefill_tokens": 0,
        "offpolicy/student_sample_tokens": 0,
        "offpolicy/student_train_tokens": 120,
        "offpolicy/teacher_prefill_tokens": 400,
        "offpolicy/teacher_cached_prefill_tokens": 90,
        "offpolicy/teacher_sample_tokens": 80,
        "offpolicy/usd": 0.5,
    }
    # A metric the backend never reported (grad_norm here) is charted as a gap,
    # not as a fabricated 0.0, and the constant phase string never lands either.
    assert "offpolicy/grad_norm" not in payload
    assert not any("phase" in key for key in payload)


def test_log_eval_uses_the_eval_namespace_and_step(fake_wandb: _FakeWandb, tmp_path: Path) -> None:
    tracker = _tracker(fake_wandb, tmp_path / "run")
    tracker.log_eval("baseline-teacher", 0.5, None)
    tracker.log_eval("step-0001", 0.75, 1)
    assert fake_wandb.log_calls == [
        ({"eval/baseline-teacher": 0.5}, 0),  # pre-training evals chart at step 0
        ({"eval/step-0001": 0.75}, 1),
    ]


def test_log_eval_charts_the_graded_companion_only_when_one_was_measured(
    fake_wandb: _FakeWandb, tmp_path: Path
) -> None:
    tracker = _tracker(fake_wandb, tmp_path / "run")
    tracker.log_eval("student-after", 0.5, 2, graded_solve_rate=0.75)
    # No graded measurement (no readable test report, or an imported older baseline): the binary
    # rate still charts, and the graded series gets a gap instead of a fabricated 0.0.
    tracker.log_eval("baseline-teacher", 0.5, None, graded_solve_rate=None)
    assert fake_wandb.log_calls == [
        ({"eval/student-after": 0.5, "eval/student-after-graded": 0.75}, 2),
        ({"eval/baseline-teacher": 0.5}, 0),
    ]


def test_log_samples_logs_a_fresh_table_under_a_step_qualified_key(
    fake_wandb: _FakeWandb, tmp_path: Path
) -> None:
    """Each call builds its own immutable table; the key carries kind and step."""
    tracker = _tracker(fake_wandb, tmp_path / "run")
    tracker.log_samples("train", 3, [_sample("t1", 1.0), _sample("t2", 0.0)])
    (payload, step) = fake_wandb.log_calls[-1]
    assert step == 3
    table = payload["samples/train-0003"]
    assert isinstance(table, _FakeTable)
    assert table.columns == ["kind", "step", "trial", "reward", "text"]
    assert table.data == [
        ["train", 3, "t1", 1.0, _sample("t1", 1.0).text],
        ["train", 3, "t2", 0.0, _sample("t2", 0.0).text],
    ]
    # A later batch logs a fresh table under its own key, never an append.
    tracker.log_samples("train", 4, [_sample("t3")])
    (payload_next, _) = fake_wandb.log_calls[-1]
    assert "samples/train-0004" in payload_next


def test_log_samples_without_a_step_charts_at_zero(fake_wandb: _FakeWandb, tmp_path: Path) -> None:
    tracker = _tracker(fake_wandb, tmp_path / "run")
    tracker.log_samples("eval-baseline-teacher", None, [_sample()])
    (payload, step) = fake_wandb.log_calls[-1]
    assert step == 0  # pre-training samples chart at step 0, like log_eval
    table = payload["samples/eval-baseline-teacher"]
    assert isinstance(table, _FakeTable)
    assert table.data[0][1] is None  # the row keeps the honest None step


def test_log_samples_with_no_samples_logs_nothing(fake_wandb: _FakeWandb, tmp_path: Path) -> None:
    tracker = _tracker(fake_wandb, tmp_path / "run")
    tracker.log_samples("train", 0, [])
    assert fake_wandb.log_calls == []


def test_log_summary_updates_the_run_summary(fake_wandb: _FakeWandb, tmp_path: Path) -> None:
    tracker = _tracker(fake_wandb, tmp_path / "run")
    tracker.log_summary(
        gate_accepted=True,
        gate_reason="student reached 100% of the teacher",
        teacher_solve_rate=0.5,
        student_before_solve_rate=0.25,
        student_after_solve_rate=0.5,
        total_usd=12.5,
        steps_completed=3,
    )
    assert fake_wandb.run.summary.updates == [
        {
            "gate_accepted": True,
            "gate_reason": "student reached 100% of the teacher",
            "teacher_solve_rate": 0.5,
            "student_before_solve_rate": 0.25,
            "student_after_solve_rate": 0.5,
            "total_usd": 12.5,
            "steps_completed": 3,
        }
    ]


def test_finish_closes_the_wandb_run(fake_wandb: _FakeWandb, tmp_path: Path) -> None:
    tracker = _tracker(fake_wandb, tmp_path / "run")
    tracker.finish()
    assert fake_wandb.finish_calls == 1


def test_a_log_failure_degrades_to_noop_with_one_warning(
    fake_wandb: _FakeWandb, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A dead dashboard must never abort a paid run: one warning, then silence."""
    tracker = _tracker(fake_wandb, tmp_path / "run")
    fake_wandb.fail_logging = True
    with caplog.at_level(logging.WARNING, logger="wmh.distill.tracking"):
        tracker.log_step(0, _metrics())  # first failure: warns, marks dead
        tracker.log_eval("step-0000", 0.5, 0)  # dead: skipped silently
        tracker.log_samples("train", 0, [_sample()])  # dead: skipped silently
        tracker.log_summary(
            gate_accepted=False,
            gate_reason="regressed",
            teacher_solve_rate=0.5,
            student_before_solve_rate=0.5,
            student_after_solve_rate=0.25,
            total_usd=1.0,
            steps_completed=1,
        )
        tracker.finish()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "training continues" in warnings[0].getMessage()
    # Nothing reached wandb after the failure, and nothing raised.
    assert fake_wandb.log_calls == []
    assert fake_wandb.run.summary.updates == []
    assert fake_wandb.finish_calls == 0
