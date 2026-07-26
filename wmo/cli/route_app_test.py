"""CLI tests for `wmo optimize route` (fit + report), driven via CliRunner."""

from __future__ import annotations

import fcntl
import importlib
import json
import os
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import pytest
from rich.console import Console
from typer.testing import CliRunner, Result

from wmo.cli.app import app
from wmo.config import HarnessConfig, save_config
from wmo.core.types import Action, ActionKind, EnvState, Observation, Session, Step, Trace
from wmo.distill.store import DistillModelCard
from wmo.engine.world_model import WorldModel
from wmo.ingest.otel_writer import write_traces_jsonl
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.policy import (
    KNN_BANK_FILENAME,
    POLICY_FILENAME,
    RoutingPolicy,
    select_model,
)
from wmo.optimize.reward import EpisodeScore
from wmo.optimize.routing import evaluate_policy
from wmo.providers import pool as pool_module
from wmo.providers.base import (
    Completion,
    Message,
    Provider,
    ProviderConfig,
    ProviderKind,
    TokenUsage,
    VerifyResult,
)
from wmo.providers.pool import PoolEntry, load_pool
from wmo.providers.registry import get_provider as registry_get_provider
from wmo.serving.traces_source import TRACES_FILENAME
from wmo.tracking import RunRecord

runner = CliRunner()


def _matrix_file(tmp_path: Path) -> Path:
    pool = [
        PoolEntry(
            name="a", kind=ProviderKind.OPENAI, model="a", input_per_mtok=1.0, output_per_mtok=1.0
        ),
        PoolEntry(
            name="b", kind=ProviderKind.OPENAI, model="b", input_per_mtok=1.0, output_per_mtok=1.0
        ),
    ]
    outcomes = []
    tasks = {
        "s1": "SELECT count(*) FROM t",
        "s2": "SELECT name FROM users WHERE id = 4",
        "s3": "write a poem about rivers",
        "s4": "draft a thank-you note",
    }
    for sid, task in tasks.items():
        sql = sid in ("s1", "s2")
        for model in ("a", "b"):
            wins = (model == "a") == sql
            outcomes.append(
                ScenarioOutcome(
                    scenario_id=sid,
                    task=task,
                    model=model,
                    reward=1.0 if wins else 0.0,
                    success=wins,
                    cost_usd=0.001,
                )
            )
    path = tmp_path / "matrix.json"
    OutcomeMatrix(pool=pool, outcomes=outcomes).save(path)
    return path


def test_route_fit_and_report(tmp_path: Path) -> None:
    matrix_file = _matrix_file(tmp_path)
    policy_file = tmp_path / "policy.json"
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(matrix_file),
            "--out",
            str(policy_file),
            "--clusters",
            "2",
            "--top-k-clusters",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    policy = RoutingPolicy.load(policy_file)
    assert policy.kind == "rank"
    assert len(policy.clusters) == 2

    report_file = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "report",
            str(matrix_file),
            str(policy_file),
            "--baseline",
            "a",
            "--out",
            str(report_file),
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(report_file.read_text())
    assert report["baseline"]["model_id"] == "a"
    assert report["headline"]["baseline_accuracy"] == 0.5
    assert report["cost_assumptions"]


def test_route_fit_rejects_unknown_embedder(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["optimize", "route", "fit", str(_matrix_file(tmp_path)), "--embedder", "vibes"],
    )
    assert result.exit_code != 0
    assert "hashing or azure" in result.output


def _knn_matrix_file(tmp_path: Path) -> Path:
    """Twelve scenarios: enough neighbors per query for a guarded fit to route at all."""
    pool = [
        PoolEntry(
            name="a", kind=ProviderKind.OPENAI, model="a", input_per_mtok=1.0, output_per_mtok=1.0
        ),
        PoolEntry(
            name="b", kind=ProviderKind.OPENAI, model="b", input_per_mtok=1.0, output_per_mtok=1.0
        ),
    ]
    sql = [
        "SELECT count(*) FROM orders WHERE total > 100",
        "SELECT name FROM users WHERE id = 4",
        "SELECT avg(price) FROM products GROUP BY category",
        "SELECT id FROM events WHERE kind = 'click'",
        "SELECT max(score) FROM matches WHERE season = 2025",
        "SELECT city FROM stores WHERE stock > 0",
    ]
    prose = [
        "write a friendly email to the team about the offsite",
        "write a warm welcome note for new employees",
        "write a short thank-you message for the organizers",
        "write a cheerful newsletter intro about spring",
        "write a gentle reminder about the expense deadline",
        "write a farewell note for a departing teammate",
    ]
    outcomes = []
    for group, tasks in (("sql", sql), ("prose", prose)):
        for index, task in enumerate(tasks):
            for model in ("a", "b"):
                wins = (model == "a") == (group == "sql")
                outcomes.append(
                    ScenarioOutcome(
                        scenario_id=f"{group}:{index}",
                        task=task,
                        model=model,
                        reward=1.0 if wins else 0.0,
                        success=wins,
                        cost_usd=0.001,
                    )
                )
    path = tmp_path / "knn_matrix.json"
    OutcomeMatrix(pool=pool, outcomes=outcomes).save(path)
    return path


def test_route_fit_knn_writes_policy_and_sidecar(tmp_path: Path) -> None:
    matrix_file = _knn_matrix_file(tmp_path)
    policy_file = tmp_path / "policy.json"
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(matrix_file),
            "--kind",
            "knn",
            "--fallback",
            "a",
            "--z",
            "0.5",
            "--rag-num",
            "3",
            "--min-pairs",
            "2",
            "--out",
            str(policy_file),
        ],
    )
    assert result.exit_code == 0, result.output
    policy = RoutingPolicy.load(policy_file)
    assert policy.kind == "knn"
    assert policy.default_model == "a" == policy.guard_model  # the pinned fallback
    assert (tmp_path / KNN_BANK_FILENAME).is_file()  # sidecar beside the policy
    assert len(policy.knn_bank().scenario_ids) == 12
    assert "routed away from the fallback" in result.output
    # The prose neighborhoods carry unanimous evidence for b, so that traffic leaves the
    # fallback while the SQL half stays on it.
    matrix = OutcomeMatrix.load(matrix_file)
    prose_ids = [sid for sid in matrix.scenario_ids() if sid.startswith("prose:")]
    assert evaluate_policy(policy, matrix, prose_ids).model_mix == {"b": 1.0}


def test_route_fit_knn_rejects_the_rank_only_cost_knob(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(_matrix_file(tmp_path)),
            "--kind",
            "knn",
            "--cost-weight",
            "0.5",
        ],
    )
    assert result.exit_code != 0
    # The message points at the knn cost control that does exist, not just at what is wrong.
    assert "--cost-quality" in result.output


def test_route_fit_rejects_unknown_kind(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["optimize", "route", "fit", str(_matrix_file(tmp_path)), "--kind", "vibes"]
    )
    assert result.exit_code != 0
    assert "knn or rank" in result.output


def _fitted_knn_policy(tmp_path: Path) -> Path:
    policy_file = tmp_path / POLICY_FILENAME
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(_knn_matrix_file(tmp_path)),
            "--kind",
            "knn",
            "--fallback",
            "a",
            "--rag-num",
            "3",
            "--min-pairs",
            "2",
            "--out",
            str(policy_file),
        ],
    )
    assert result.exit_code == 0, result.output
    return policy_file


def test_route_tune_sets_the_dial_and_keeps_the_policy_as_fitted(tmp_path: Path) -> None:
    policy_file = _fitted_knn_policy(tmp_path)
    fitted = RoutingPolicy.load(policy_file)
    result = runner.invoke(
        app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "0.6"]
    )
    assert result.exit_code == 0, result.output
    tuned = RoutingPolicy.load(policy_file)
    assert tuned.cost_quality == 0.6
    assert tuned.pick_lam > 0.0
    assert tuned.guard_mode == "asymmetric"
    # The un-tuned artifact is preserved, so the dial is always re-appliable from the fit.
    base = RoutingPolicy.load(tmp_path / "policy.base.json")
    assert base.model_dump() == fitted.model_dump()
    # The printed anchor table is how an operator learns what the position measured.
    assert "cost_quality=0.6" in result.output
    assert "-46.2%" in result.output


def test_route_tune_twice_equals_tuning_once_from_the_base(tmp_path: Path) -> None:
    policy_file = _fitted_knn_policy(tmp_path)
    runner.invoke(app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "1.0"])
    once = RoutingPolicy.load(policy_file).model_dump()
    for _ in range(2):
        result = runner.invoke(
            app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "1.0"]
        )
        assert result.exit_code == 0, result.output
    assert RoutingPolicy.load(policy_file).model_dump() == once
    # Sliding back down lands exactly where a first-time slide to that position would.
    runner.invoke(app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "0.25"])
    balanced = RoutingPolicy.load(policy_file)
    assert balanced.cost_quality == 0.25
    assert (balanced.pick_lam, balanced.guard_mode) == (0.0, "symmetric")


def test_route_tune_still_routes_after_the_dial_moves(tmp_path: Path) -> None:
    # The dial must leave a servable policy: same bank, same baseline, still routing.
    matrix_file = _knn_matrix_file(tmp_path)
    policy_file = _fitted_knn_policy(tmp_path)
    runner.invoke(app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "1.0"])
    tuned = RoutingPolicy.load(policy_file)
    matrix = OutcomeMatrix.load(matrix_file)
    prose_ids = [sid for sid in matrix.scenario_ids() if sid.startswith("prose:")]
    assert evaluate_policy(tuned, matrix, prose_ids).model_mix == {"b": 1.0}


def test_route_tune_rejects_a_policy_kind_without_a_dial(tmp_path: Path) -> None:
    policy_file = tmp_path / POLICY_FILENAME
    fit = runner.invoke(
        app,
        ["optimize", "route", "fit", str(_matrix_file(tmp_path)), "--out", str(policy_file)],
    )
    assert fit.exit_code == 0, fit.output
    result = runner.invoke(
        app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "0.5"]
    )
    assert result.exit_code != 0
    assert "kind='rank'" in result.output


def test_route_tune_rejects_a_missing_policy_file(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["optimize", "route", "tune", str(tmp_path / "nope.json"), "--cost-quality", "0.5"]
    )
    assert result.exit_code != 0
    assert "no policy file" in result.output


def test_route_tune_rejects_a_dial_outside_the_range(tmp_path: Path) -> None:
    policy_file = _fitted_knn_policy(tmp_path)
    result = runner.invoke(
        app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "2"]
    )
    assert result.exit_code != 0


def _run_dir(tmp_path: Path, sampler: str = "tinker://fake/sampler/final/0") -> Path:
    """A distillation run dir with just the artifact `route student` reads."""
    run_dir = tmp_path / "distill" / "support"
    run_dir.mkdir(parents=True, exist_ok=True)
    card = DistillModelCard(
        base_model="Qwen/Qwen3-8B",
        lora_rank=32,
        teacher_model="glm-5.2",
        sampler_path=sampler,
        steps_completed=200,
    )
    (run_dir / "model_card.json").write_text(card.model_dump_json(indent=2), encoding="utf-8")
    return run_dir


def _built_model(tmp_path: Path, name: str = "support") -> Path:
    """A world model dir as `WorldModelStore` recognizes one (a dir carrying config.toml)."""
    model_dir = tmp_path / "models" / name
    model_dir.mkdir(parents=True)
    (model_dir / "config.toml").write_text("", encoding="utf-8")
    return model_dir


def _add_student(tmp_path: Path, pool_file: Path, *, name: str = "student") -> Result:
    return runner.invoke(
        app,
        [
            "optimize",
            "route",
            "student",
            str(_run_dir(tmp_path)),
            "--input-per-mtok",
            "0.1",
            "--output-per-mtok",
            "0.4",
            "--name",
            name,
            "--pool",
            str(pool_file),
        ],
    )


def test_route_student_makes_a_trained_adapter_routable(tmp_path: Path) -> None:
    """The keystone: a run dir becomes a loadable pool candidate with no hand-edited TOML."""
    pool_file = tmp_path / "pool.toml"

    result = _add_student(tmp_path, pool_file)

    assert result.exit_code == 0, result.output
    entry = load_pool(pool_file).entry("student")
    assert entry.kind is ProviderKind.OPENAI
    assert entry.model == "tinker://fake/sampler/final/0"
    assert entry.model_type == "Qwen/Qwen3-8B"
    assert entry.chat_max_tokens_field == "max_tokens"
    assert entry.api_key_env == "TINKER_API_KEY"
    assert entry.price().input_per_mtok == 0.1


def test_route_student_requires_prices(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["optimize", "route", "student", str(_run_dir(tmp_path)), "--pool", str(tmp_path / "p")],
    )
    assert result.exit_code != 0
    assert "--input-per-mtok" in result.output


def test_route_student_names_the_missing_model_card(tmp_path: Path) -> None:
    empty = tmp_path / "not-a-run"
    empty.mkdir()
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "student",
            str(empty),
            "--input-per-mtok",
            "0.1",
            "--output-per-mtok",
            "0.4",
        ],
    )
    assert result.exit_code != 0
    assert "model_card.json" in result.output
    assert "adapter version directory" in result.output  # says what to pass instead


def test_route_student_rejects_a_run_with_no_trained_weights(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path, sampler="Qwen/Qwen3-8B")
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "student",
            str(run_dir),
            "--input-per-mtok",
            "0.1",
            "--output-per-mtok",
            "0.4",
        ],
    )
    assert result.exit_code != 0
    assert "tinker://" in result.output


def test_route_student_declining_the_replacement_leaves_the_pool_alone(tmp_path: Path) -> None:
    pool_file = tmp_path / "pool.toml"
    assert _add_student(tmp_path, pool_file).exit_code == 0
    before = pool_file.read_text(encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "student",
            str(_run_dir(tmp_path)),
            "--input-per-mtok",
            "9.9",
            "--output-per-mtok",
            "9.9",
            "--pool",
            str(pool_file),
        ],
        input="n\n",
    )

    assert result.exit_code == 0
    assert pool_file.read_text(encoding="utf-8") == before  # the 9.9 price never landed


def test_route_student_replaces_the_same_name_under_yes(tmp_path: Path) -> None:
    pool_file = tmp_path / "pool.toml"
    assert _add_student(tmp_path, pool_file).exit_code == 0

    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "student",
            str(_run_dir(tmp_path)),
            "--input-per-mtok",
            "0.2",
            "--output-per-mtok",
            "0.8",
            "--pool",
            str(pool_file),
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "replaced" in result.output
    models = load_pool(pool_file).models
    assert len(models) == 1  # replaced, not duplicated
    assert models[0].price().output_per_mtok == 0.8


def test_route_pin_writes_a_serveable_static_policy(tmp_path: Path) -> None:
    """One step from pool candidate to endpoint: the policy lands where `wmo serve` reads it."""
    pool_file = tmp_path / "pool.toml"
    assert _add_student(tmp_path, pool_file).exit_code == 0
    model_dir = _built_model(tmp_path)

    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "pin",
            "support",
            "--model",
            "student",
            "--pool",
            str(pool_file),
            "--root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    policy = RoutingPolicy.load(model_dir / POLICY_FILENAME)
    assert policy.kind == "static"
    assert policy.default_model == "student"
    assert [entry.name for entry in policy.pool] == ["student"]
    assert policy.fitted_from is not None
    assert "no outcome matrix" in policy.fitted_from  # provenance says it measured nothing


def test_route_pin_serves_through_the_endpoint_it_installed(tmp_path: Path) -> None:
    """The pinned policy is not just well formed: `select_model` actually routes on it."""
    pool_file = tmp_path / "pool.toml"
    assert _add_student(tmp_path, pool_file).exit_code == 0
    model_dir = _built_model(tmp_path)
    runner.invoke(
        app,
        [
            "optimize",
            "route",
            "pin",
            "support",
            "--model",
            "student",
            "--pool",
            str(pool_file),
            "--root",
            str(tmp_path),
        ],
    )

    policy = RoutingPolicy.load(model_dir / POLICY_FILENAME)
    decision = select_model(policy, "anything at all")

    assert decision.model == "student"


def test_route_pin_rejects_a_model_outside_the_pool(tmp_path: Path) -> None:
    pool_file = tmp_path / "pool.toml"
    assert _add_student(tmp_path, pool_file).exit_code == 0
    _built_model(tmp_path)

    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "pin",
            "support",
            "--model",
            "ghost",
            "--pool",
            str(pool_file),
            "--root",
            str(tmp_path),
        ],
    )

    assert result.exit_code != 0
    assert "no pool model named 'ghost'" in result.output
    assert "student" in result.output  # lists what IS available


def test_route_pin_names_the_missing_world_model(tmp_path: Path) -> None:
    pool_file = tmp_path / "pool.toml"
    assert _add_student(tmp_path, pool_file).exit_code == 0

    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "pin",
            "nope",
            "--model",
            "student",
            "--pool",
            str(pool_file),
            "--root",
            str(tmp_path),
        ],
    )

    assert result.exit_code != 0
    assert "no world model named 'nope'" in result.output


def test_route_pin_declining_keeps_a_fitted_policy(tmp_path: Path) -> None:
    """Pinning over a fitted knn policy would orphan its evidence bank, so it must ask first."""
    pool_file = tmp_path / "pool.toml"
    assert _add_student(tmp_path, pool_file).exit_code == 0
    model_dir = _built_model(tmp_path)
    installed = model_dir / POLICY_FILENAME
    fitted = _fitted_knn_policy(tmp_path)
    installed.write_text(fitted.read_text(encoding="utf-8"), encoding="utf-8")
    before = installed.read_text(encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "pin",
            "support",
            "--model",
            "student",
            "--pool",
            str(pool_file),
            "--root",
            str(tmp_path),
        ],
        input="n\n",
    )

    assert result.exit_code == 0
    assert installed.read_text(encoding="utf-8") == before


def test_route_student_rejects_an_empty_endpoint(tmp_path: Path) -> None:
    """`--endpoint "$UNSET_VAR"` must not silently fall back to a different host."""
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "student",
            str(_run_dir(tmp_path)),
            "--input-per-mtok",
            "0.1",
            "--output-per-mtok",
            "0.4",
            "--endpoint",
            "",
            "--pool",
            str(tmp_path / "pool.toml"),
        ],
    )

    assert result.exit_code != 0
    assert "--endpoint is empty" in result.output
    assert not (tmp_path / "pool.toml").exists()  # nothing was written


def test_route_student_summary_does_not_claim_a_key_it_will_not_send(tmp_path: Path) -> None:
    """A custom endpoint authenticates via WMO_ENDPOINT_API_KEY, and the summary must say so."""
    pool_file = tmp_path / "pool.toml"
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "student",
            str(_run_dir(tmp_path)),
            "--input-per-mtok",
            "0.1",
            "--output-per-mtok",
            "0.4",
            "--endpoint",
            "https://my-vllm.example/v1",
            "--pool",
            str(pool_file),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "WMO_ENDPOINT_API_KEY" in result.output
    assert "TINKER_API_KEY" not in result.output
    assert load_pool(pool_file).entry("student").api_key_env is None


def test_route_student_reports_a_busy_pool_without_claiming_it_registered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A held roster lock is a retryable busy state, so it must exit non-zero and say to retry.

    The lock is real here (taken with flock, from this process, on the file the command writes);
    only the wait is shortened, so the CLI runs the same path an operator hits when a second
    registration is in flight. What matters is that it never prints its "added pool candidate"
    line for a write that did not happen, and never reports a lock holder as a bad flag.
    """
    monkeypatch.setattr(pool_module, "POOL_LOCK_TIMEOUT_S", 0.05)
    pool_file = tmp_path / "pool.toml"
    lock_path = pool_file.with_name(f"{pool_file.name}.lock")
    pool_file.parent.mkdir(parents=True, exist_ok=True)
    holder = os.open(lock_path, os.O_CREAT | os.O_WRONLY, 0o600)
    fcntl.flock(holder, fcntl.LOCK_EX)
    try:
        result = runner.invoke(
            app,
            [
                "optimize",
                "route",
                "student",
                str(_run_dir(tmp_path)),
                "--input-per-mtok",
                "0.1",
                "--output-per-mtok",
                "0.4",
                "--pool",
                str(pool_file),
            ],
        )
    finally:
        os.close(holder)

    assert result.exit_code == 1, result.output
    assert "pool busy" in result.output
    assert "retry" in result.output
    assert "added pool candidate" not in result.output
    assert not pool_file.exists()  # nothing was written


def test_route_student_rejects_an_unknown_output_budget_field(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "student",
            str(_run_dir(tmp_path)),
            "--input-per-mtok",
            "0.1",
            "--output-per-mtok",
            "0.4",
            "--chat-max-tokens-field",
            "max_output_tokens",
            "--pool",
            str(tmp_path / "pool.toml"),
        ],
    )

    assert result.exit_code != 0
    assert "max_tokens or max_completion_tokens" in result.output


route_module = importlib.import_module("wmo.cli.route_app")


_HELD_OUT_IDS = ("tr-010", "tr-018", "tr-020", "tr-027")


_HELD_OUT_TOOL = "holdout_only"


def _corpus(count: int = 30) -> list[Trace]:
    """A trace corpus whose split has a real held-out band, one task prompt per trace.

    Emitted in DESCENDING id order on purpose: the sweep sorts the held-out band by trace id
    before applying `--scenarios`, and a corpus already in id order would make that sort
    unobservable (any assertion on which prefix was cut would hold without it).
    """
    traces: list[Trace] = []
    for index in reversed(range(count)):
        trace_id = f"tr-{index:03d}"
        # Held-out traces call a tool no train trace does, so a tools hint derived from the wrong
        # band is visible in the candidate's system prompt.
        tool = _HELD_OUT_TOOL if trace_id in _HELD_OUT_IDS else "ls"
        traces.append(
            Trace(
                trace_id=trace_id,
                steps=[
                    Step(
                        action=Action(
                            kind=ActionKind.TOOL_CALL, name=tool, arguments={"path": "."}
                        ),
                        observation=Observation(content="a.txt"),
                        task=f"task {trace_id}",
                    ),
                    Step(
                        action=Action(kind=ActionKind.MESSAGE, content="done"),
                        observation=Observation(content="ok"),
                        task=f"task {trace_id}",
                    ),
                ],
            )
        )
    return traces


def _project(tmp_path: Path, *, traces: list[Trace] | None, train_split: float = 0.8) -> Path:
    """Write a minimal built-model artifact (config + its own corpus) and return the root."""
    root = tmp_path / ".wmo"
    model_dir = root / "models" / "support"
    save_config(
        HarnessConfig(
            providers=[ProviderConfig(kind=ProviderKind.ANTHROPIC, model="fake-serve")],
            serve_provider=ProviderKind.ANTHROPIC,
            train_split=train_split,
        ),
        model_dir,
    )
    if traces is not None:
        write_traces_jsonl(traces, model_dir / TRACES_FILENAME)
    return root


def _pool_file(tmp_path: Path) -> Path:
    """Two priced candidates: 1/2 and 10/20 USD per Mtok, so cost lines are distinguishable."""
    path = tmp_path / "pool.toml"
    path.write_text(
        "[[model]]\n"
        'name = "cheap"\n'
        'kind = "openai"\n'
        'model = "cheap-1"\n'
        "input_per_mtok = 1.0\n"
        "output_per_mtok = 2.0\n"
        "\n"
        "[[model]]\n"
        'name = "pricey"\n'
        'kind = "openai"\n'
        'model = "pricey-1"\n'
        "input_per_mtok = 10.0\n"
        "output_per_mtok = 20.0\n",
        encoding="utf-8",
    )
    return path


class _FakeWorldModel:
    """`WorldModel`-shaped stub: in-memory sessions, a canned episode score, no LLM at all."""

    def __init__(self, reward: float = 0.75, judge_fails_on: frozenset[str] = frozenset()) -> None:
        self._reward = reward
        self._frozen = False
        # Tasks whose judge call raises, for every candidate: a scenario the whole pool loses.
        self._judge_fails_on = judge_fails_on
        self._task_of: dict[str, str | None] = {}
        self.tasks: list[str | None] = []
        self.scored: list[str] = []
        self.opened_frozen: list[bool] = []  # was index enrichment suspended for this episode?

    @contextmanager
    def frozen(self) -> Iterator[_FakeWorldModel]:
        self._frozen = True
        try:
            yield self
        finally:
            self._frozen = False

    def new_session(
        self, task: str | None = None, seed_state: EnvState | None = None, *, enrich: bool = True
    ) -> Session:
        self.tasks.append(task)
        self.opened_frozen.append(self._frozen)
        session = Session(id=f"s{len(self.tasks)}", task=task, enrich=enrich)
        self._task_of[session.id] = task
        return session

    def step(self, session_id: str, action: Action) -> Observation:
        return Observation(content="ok")

    def score_session(self, session_id: str) -> EpisodeScore:
        task = self._task_of.get(session_id)
        if task is not None and task in self._judge_fails_on:
            # WorldModelEnv.close preserves this and `last_score` re-raises it, which is how a
            # throttled judge leaves a cell unscored for every candidate that ran the scenario.
            raise RuntimeError("judge throttled (429)")
        self.scored.append(session_id)
        return EpisodeScore(reward=self._reward, success=True, critique="fine")

    def end_session(self, session_id: str) -> RunRecord:
        return RunRecord(run_id=session_id, kind="serve")

    def session_usage(self, session_id: str) -> RunRecord:
        return RunRecord(run_id=session_id, kind="serve")


class _ScriptedCandidate:
    """A candidate model that calls one tool and then declares itself done.

    `throttled` makes every completion raise instead, the way a rate-limited candidate does: the
    episode errors, `run_episode` records it, and the cell comes back unscored.
    """

    def __init__(
        self, config: ProviderConfig, systems: list[str], *, throttled: bool = False
    ) -> None:
        self.config = config
        self._systems = systems
        self._throttled = throttled
        self._script = ['{"tool": "ls", "arguments": {}}', '{"done": true, "summary": "ok"}']
        self._index = 0

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        self._systems.append(system)
        if self._throttled:
            raise RuntimeError("rate limit exceeded (429)")
        text = self._script[min(self._index, len(self._script) - 1)]
        self._index += 1
        return Completion(text=text, usage=TokenUsage(input_tokens=10, output_tokens=5))

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def verify(self) -> VerifyResult:
        raise NotImplementedError


class _Answer:
    """A `rich.prompt.Confirm` stand-in that always answers the same way."""

    def __init__(self, answer: bool) -> None:
        self._answer = answer

    def ask(self, prompt: str, *, default: bool = True) -> bool:
        return self._answer


class _Seams:
    """What the sweep's two stubbed seams recorded, for post-run assertions."""

    def __init__(self, world_model: _FakeWorldModel) -> None:
        self.world_model = world_model
        self.built_providers: list[str] = []  # one entry per candidate provider constructed
        self.systems: list[str] = []  # every system prompt a candidate was called with


def _patch_seams(
    monkeypatch: pytest.MonkeyPatch,
    *,
    reward: float = 0.75,
    no_scoring: bool = False,
    throttled_models: frozenset[str] = frozenset(),
    throttled_episodes: dict[str, tuple[bool, ...]] | None = None,
    judge_fails_on: frozenset[str] = frozenset(),
    real_kinds: frozenset[ProviderKind] = frozenset(),
) -> _Seams:
    """Stub the world model and the pool's provider construction; return the recorder.

    Args:
        monkeypatch: The patcher whose lifetime the stubs live for.
        reward: Reward the fake judge returns for every scored episode.
        no_scoring: Build the env WITHOUT `score_on_close`, which is what the pre-change code path
            would amount to: it exists so a test can show the difference is observable.
        throttled_models: Provider model ids (`PoolEntry.model`) whose completions raise, so that
            candidate's cells come back unscored while the others are scored.
        throttled_episodes: Per-model cycle of throttled flags over each scenario's episodes:
            `{"pricey-1": (False, True, True)}` keeps episode 0 and loses episodes 1 and 2 of EVERY
            scenario. `evaluate_pool` builds one provider per cell and a candidate meets its own
            cells scenario by scenario, episode by episode (whatever the other candidates are
            doing between them), so counting THIS model's constructions gives the episode index
            within the scenario. This is how a candidate comes back with the same scenarios as
            the others but FEWER scored episodes on them.
        judge_fails_on: Scenario tasks whose episode SCORING raises, for every candidate: the
            whole pool loses those scenarios together.
        real_kinds: Provider kinds to construct for real instead of faking, so a test can exercise
            a backend that refuses its own config or cannot build its lazy client. Construction and
            preparation are both request-free, and no real provider is ever called (the sweep must
            fail before any cell runs).
    """
    seams = _Seams(_FakeWorldModel(reward=reward, judge_fails_on=judge_fails_on))
    episode_cycles = throttled_episodes or {}

    def _load(model_dir: Path) -> tuple[WorldModel, Provider]:
        provider = _ScriptedCandidate(
            ProviderConfig(kind=ProviderKind.ANTHROPIC, model="fake-serve"), []
        )
        return cast("WorldModel", seams.world_model), cast("Provider", provider)

    def _get_provider(config: ProviderConfig, api_key: str | None = None) -> Provider:
        if config.kind in real_kinds:
            return registry_get_provider(config, api_key=api_key)
        seams.built_providers.append(config.model)
        cycle = episode_cycles.get(config.model)
        throttled = config.model in throttled_models
        if cycle:
            # The pre-flight builds one provider per candidate before any episode runs, so the
            # sweep's first episode is this model's SECOND construction; from there the count is
            # the episode index (`evaluate_pool` builds one provider per episode).
            episode = seams.built_providers.count(config.model) - 2
            throttled = throttled or (episode >= 0 and cycle[episode % len(cycle)])
        return cast(
            "Provider",
            _ScriptedCandidate(config, seams.systems, throttled=throttled),
        )

    monkeypatch.setattr(route_module, "load_world_model", _load)
    monkeypatch.setattr("wmo.providers.pool.get_provider", _get_provider)
    if no_scoring:
        real = route_module.WorldModelEnv
        monkeypatch.setattr(
            route_module,
            "WorldModelEnv",
            lambda world_model, *, score_on_close=False: real(world_model),
        )
    return seams


def _sweep(tmp_path: Path, root: Path, *extra: str) -> tuple[Path, Result]:
    """Invoke `wmo optimize route sweep` against the temp project; return (out path, result)."""
    out = tmp_path / "matrix.json"
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "sweep",
            "--root",
            str(root),
            "--pool",
            str(_pool_file(tmp_path)),
            "--out",
            str(out),
            *extra,
        ],
    )
    return out, result


_FRAME_CHARS = frozenset("│┃╭╮╰╯─━┏┓┗┛┡┩┢┪╇╈├┤┬┴┼")


def _flat(text: str) -> str:
    """Text with whitespace and rich's box-drawing frame removed.

    Typer renders a usage error inside a rich panel and the cost estimate inside a rich table,
    both of which wrap (and frame) long values, so a literal substring check against the raw
    output is a coin flip on where the wrap landed. Assert against this instead.
    """
    return "".join(ch for ch in text if not ch.isspace() and ch not in _FRAME_CHARS)


def _says(result_output: str, phrase: str) -> bool:
    """Whether the CLI said `phrase`, ignoring wherever rich wrapped the line."""
    return _flat(phrase) in _flat(result_output)


def test_route_sweep_writes_a_matrix_fit_can_consume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seams = _patch_seams(monkeypatch)
    root = _project(tmp_path, traces=_corpus())
    out, result = _sweep(tmp_path, root, "support", "--scenarios", "3", "--max-steps", "4", "--yes")
    assert result.exit_code == 0, result.output

    matrix = OutcomeMatrix.load(out)
    # Leak-free AND deterministic: the first three held-out traces by id, never a train task.
    assert matrix.scenario_ids() == list(_HELD_OUT_IDS[:3])
    assert {o.task for o in matrix.outcomes} == {f"task {tid}" for tid in _HELD_OUT_IDS[:3]}
    assert matrix.model_names() == ["cheap", "pricey"]
    assert len(matrix.outcomes) == 6  # 2 candidates x 3 scenarios x 1 episode
    assert all(o.reward == 0.75 for o in matrix.outcomes)
    assert matrix.mean_reward("cheap") == 0.75
    # The candidates saw the corpus's tool surface, summarized from the TRAIN split only: the
    # held-out band's own tool never reaches them (deriving the hint from the measured band would
    # leak, and would make the hint depend on where `--scenarios` cut).
    assert seams.systems
    assert all("ls(path)" in system for system in seams.systems)
    assert not any(_HELD_OUT_TOOL in system for system in seams.systems)
    # Progress streamed per cell, and the printed handoff chains the workflow.
    assert "[1/6]" in result.output and "[6/6]" in result.output
    assert _says(result.output, f"wmo optimize route fit {out} --kind knn")
    # Per-candidate scored counts print even on a clean sweep, so "3 of 3 each" is visible rather
    # than inferred from a total that cannot distinguish 3+3 from 5+1.
    assert _says(result.output, "Scored coverage per candidate")
    assert _flat(result.output).count("cheap30-") == 1  # cheap: 3 scored, 0 unscored, none lost

    # The whole point of the matrix: `fit` consumes it without further preparation.
    policy_file = tmp_path / "policy.json"
    fitted = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(out),
            "--out",
            str(policy_file),
            "--clusters",
            "2",
            "--top-k-clusters",
            "1",
        ],
    )
    assert fitted.exit_code == 0, fitted.output
    assert RoutingPolicy.load(policy_file).kind == "rank"


def test_route_sweep_scores_every_episode_through_a_scoring_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `WorldModelEnv(..., score_on_close=True)` is the contract `evaluate_pool` needs: the env
    # must judge each episode as it closes, or no cell is evidence.
    seams = _patch_seams(monkeypatch, reward=0.5)
    root = _project(tmp_path, traces=_corpus())
    out, result = _sweep(tmp_path, root, "support", "--scenarios", "2", "--yes")
    assert result.exit_code == 0, result.output
    matrix = OutcomeMatrix.load(out)
    assert len(seams.world_model.scored) == len(matrix.outcomes) == 4
    assert all(o.scored and o.error is None and o.success for o in matrix.outcomes)
    assert [o.reward for o in matrix.outcomes] == [0.5] * 4
    # Every episode ran with index enrichment suspended, so no candidate's predictions can
    # become the next candidate's retrieved demos (which would make scores sweep-order dependent).
    assert seams.world_model.opened_frozen == [True] * 4


def test_route_sweep_without_a_scoring_env_produces_no_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The negative control for the test above: drop `score_on_close` and every cell comes back
    # unscored, so the failure the assertion catches is a real behavioral difference.
    seams = _patch_seams(monkeypatch, no_scoring=True)
    root = _project(tmp_path, traces=_corpus())
    out, result = _sweep(tmp_path, root, "support", "--scenarios", "2", "--yes")
    # A sweep that scored nothing exits NON-ZERO, so `sweep && fit` in a script stops instead of
    # fitting on a matrix the command itself says is not evidence.
    assert result.exit_code == 1, result.output
    assert seams.world_model.scored == []
    # ... and the paid rows are still on disk, carrying the `error` that explains them.
    matrix = OutcomeMatrix.load(out)
    assert all(not o.scored for o in matrix.outcomes)
    assert _says(result.output, "no cell was scored")


def test_route_sweep_withholds_the_fit_handoff_on_uneven_scored_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `pricey` is throttled on every call, so its episodes error and its cells go UNSCORED while
    # `cheap` is scored on all three scenarios. Both fitters SKIP unscored rows, so ranking these
    # two against each other would compare 3 scenarios of cheap with 0 of pricey.
    seams = _patch_seams(monkeypatch, throttled_models=frozenset({"pricey-1"}))
    root = _project(tmp_path, traces=_corpus())
    out, result = _sweep(tmp_path, root, "support", "--scenarios", "3", "--yes")
    assert result.exit_code == 1, result.output
    # The artifact is still written: those cells were paid for, and their `error` is the diagnosis.
    matrix = OutcomeMatrix.load(out)
    assert [o.scored for o in matrix.outcomes if o.model == "cheap"] == [True] * 3
    assert [o.scored for o in matrix.outcomes if o.model == "pricey"] == [False] * 3
    assert all("429" in (o.error or "") for o in matrix.outcomes if o.model == "pricey")
    flat = _flat(result.output)
    # The user can see WHICH candidate lost WHICH scenarios, not just a total.
    assert _says(result.output, "Scored coverage per candidate")
    assert _says(result.output, ", ".join(_HELD_OUT_IDS[:3]))
    assert "DIFFERENTscenarios" in flat and "cheap3,pricey0" in flat
    # A candidate the coverage table can only show as all-zero gets its cause quoted from the
    # matrix, named, with what to do: it is the one failure the table itself cannot explain.
    assert _says(result.output, "pricey was never scored; its first cell failed with")
    assert "ratelimitexceeded(429)" in flat
    assert _says(result.output, "fix that entry in the pool file")
    # The handoff is withheld, so `sweep && fit` in a script stops instead of fitting on it, and
    # the message names the one flag that proceeds anyway.
    assert "wmooptimizeroutefit" not in flat
    assert "--allow-uneven-coverage" in flat
    # A real sweep, not an aborted one: every cell ran (the cells are the paid evidence).
    assert len(seams.world_model.tasks) == 6


def test_route_sweep_calls_out_a_first_cell_failure_while_the_rest_can_still_be_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The residual the pre-flight cannot close: a bedrock candidate's missing AWS credentials are
    # only knowable over the wire (measured: boto3 reaches the instance-metadata endpoint and
    # builds a client with no credentials at all), so that entry passes the pre-flight and dies at
    # its FIRST CELL. `pricey` stands in for it. What must not happen is the operator learning
    # only after `cheap` has run the whole scenario set: cells run candidate-minor, so the failure
    # is cell 2 of 8 and is named there, with six cells of spend still recoverable by Ctrl-C.
    _patch_seams(monkeypatch, throttled_models=frozenset({"pricey-1"}))
    root = _project(tmp_path, traces=_corpus())
    _out, result = _sweep(tmp_path, root, "support", "--scenarios", "4", "--yes")
    assert result.exit_code == 1, result.output  # uneven coverage, so no fit handoff either
    flat = _flat(result.output)
    assert "theFIRSTcellofpriceyfailed" in flat
    assert "ratelimitexceeded(429)" in flat  # what broke, quoted from the cell itself
    assert _says(result.output, "stop with Ctrl-C, fix its pool entry, and sweep again")
    # Before the third cell ever ran, which is the whole point of the ordering.
    assert flat.index("theFIRSTcellofpriceyfailed") < flat.index("[3/8]")
    # Once per candidate, not once per lost cell: pricey loses all four and is called out once.
    assert flat.count("theFIRSTcellofpriceyfailed") == 1


def test_route_sweep_allow_uneven_coverage_hands_off_and_still_states_the_bias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The opt-out for an operator who knows a candidate's backend was down all sweep and wants the
    # partial data anyway: same coverage table, same warning, but the handoff stands.
    _patch_seams(monkeypatch, throttled_models=frozenset({"pricey-1"}))
    root = _project(tmp_path, traces=_corpus())
    out, result = _sweep(
        tmp_path, root, "support", "--scenarios", "3", "--yes", "--allow-uneven-coverage"
    )
    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert "DIFFERENTscenarios" in flat and "biasaccepted" in flat
    assert _says(result.output, f"wmo optimize route fit {out} --kind knn")


def test_route_sweep_withholds_the_fit_handoff_on_uneven_scored_episodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Uneven EPISODES, not uneven scenario presence: `pricey` keeps episode 0 of every scenario and
    # loses episodes 1 and 2, so both candidates cover BOTH scenarios and a presence-only gate sees
    # nothing wrong. What differs is the scored episode COUNT per (candidate, scenario), which is
    # exactly what the fitters weigh: `fit_rank_policy` averages every surviving episode into its
    # cluster mean (so a scenario counts three times as much for `cheap` as for `pricey`), and
    # `_overall_best` / `best_single_on_fit` pick the default and knn fallback off the same
    # episode-weighted means. Which episodes happened to fail must not decide the policy.
    seams = _patch_seams(monkeypatch, throttled_episodes={"pricey-1": (False, True, True)})
    root = _project(tmp_path, traces=_corpus())
    out, result = _sweep(tmp_path, root, "support", "--scenarios", "2", "--episodes", "3", "--yes")
    assert result.exit_code == 1, result.output
    # Every cell ran and the artifact is on disk: 2 candidates x 2 scenarios x 3 episodes.
    matrix = OutcomeMatrix.load(out)
    assert len(matrix.outcomes) == 12
    assert len(seams.world_model.tasks) == 12
    assert Counter(
        (outcome.model, outcome.scenario_id) for outcome in matrix.outcomes if outcome.scored
    ) == Counter(
        {
            ("cheap", _HELD_OUT_IDS[0]): 3,
            ("cheap", _HELD_OUT_IDS[1]): 3,
            ("pricey", _HELD_OUT_IDS[0]): 1,
            ("pricey", _HELD_OUT_IDS[1]): 1,
        }
    )
    # Presence is identical for both candidates, so nothing but the counts could have caught this.
    scored_scenarios = {
        name: {o.scenario_id for o in matrix.outcomes if o.scored and o.model == name}
        for name in ("cheap", "pricey")
    }
    assert scored_scenarios["cheap"] == scored_scenarios["pricey"] == set(_HELD_OUT_IDS[:2])
    flat = _flat(result.output)
    assert _says(result.output, "DIFFERENT numbers of scored episodes")
    # The table says WHICH candidate thinned WHICH scenario, and by how much.
    assert "pricey24" in flat  # 2 scored cells, 4 unscored
    assert _says(result.output, f"{_HELD_OUT_IDS[0]} 1/3")
    assert _says(result.output, f"{_HELD_OUT_IDS[1]} 1/3")
    # The handoff is withheld, so `sweep && fit` stops here, and the one opt-out is named.
    assert "wmooptimizeroutefit" not in flat
    assert "--allow-uneven-coverage" in flat


def test_route_sweep_allow_uneven_coverage_also_covers_uneven_episodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Same opt-out, same bias statement, for the episode-count case: an operator who knows one
    # candidate was throttled through part of the sweep and wants the partial data anyway.
    _patch_seams(monkeypatch, throttled_episodes={"pricey-1": (False, True, True)})
    root = _project(tmp_path, traces=_corpus())
    out, result = _sweep(
        tmp_path,
        root,
        "support",
        "--scenarios",
        "2",
        "--episodes",
        "3",
        "--yes",
        "--allow-uneven-coverage",
    )
    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert _says(result.output, "DIFFERENT numbers of scored episodes")
    assert "biasaccepted" in flat
    assert _says(result.output, f"wmo optimize route fit {out} --kind knn")


def test_route_sweep_hands_off_when_every_candidate_lost_the_same_episodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The negative control for the two tests above: EVERY candidate loses episodes 1 and 2 of every
    # scenario, so the per-(candidate, scenario) counts are identical. That is still a comparison,
    # like-for-like on one episode per scenario, so the handoff stands and only the counts show it.
    _patch_seams(
        monkeypatch,
        throttled_episodes={"cheap-1": (False, True, True), "pricey-1": (False, True, True)},
    )
    root = _project(tmp_path, traces=_corpus())
    out, result = _sweep(tmp_path, root, "support", "--scenarios", "2", "--episodes", "3", "--yes")
    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert "cheap24-" in flat and "pricey24-" in flat  # 2 scored, 4 unscored, nothing thinner
    assert "DIFFERENT" not in flat
    assert _says(result.output, f"wmo optimize route fit {out} --kind knn")


def test_route_sweep_hands_off_when_every_candidate_lost_the_same_scenario(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The judge throttles on ONE scenario, so every candidate loses that scenario and none of the
    # others. Coverage stays like-for-like on what is left, which IS a comparison: the handoff
    # stands, and the counts still show the loss.
    lost = f"task {_HELD_OUT_IDS[1]}"
    _patch_seams(monkeypatch, judge_fails_on=frozenset({lost}))
    root = _project(tmp_path, traces=_corpus())
    out, result = _sweep(tmp_path, root, "support", "--scenarios", "3", "--yes")
    assert result.exit_code == 0, result.output
    matrix = OutcomeMatrix.load(out)
    unscored = [o for o in matrix.outcomes if not o.scored]
    assert {o.model for o in unscored} == {"cheap", "pricey"}
    assert {o.scenario_id for o in unscored} == {_HELD_OUT_IDS[1]}
    flat = _flat(result.output)
    assert f"cheap21{_HELD_OUT_IDS[1]}" in flat  # 2 scored, 1 unscored, and which one it lost
    assert "DIFFERENTscenarios" not in flat
    assert _says(result.output, f"wmo optimize route fit {out} --kind knn")


def test_route_sweep_declining_the_confirmation_spends_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seams = _patch_seams(monkeypatch)
    root = _project(tmp_path, traces=_corpus())
    monkeypatch.setattr(route_module, "_console", Console(force_terminal=True))
    monkeypatch.setattr(route_module, "Confirm", _Answer(False))
    out, result = _sweep(tmp_path, root, "support", "--scenarios", "3")
    assert result.exit_code == 0, result.output
    assert not out.exists()  # nothing written
    # The pre-flight DID construct both candidates (that is how an unusable backend becomes a usage
    # error before the cost question), but construction is side-effect free: what proves nothing
    # was spent is that no candidate was ever CALLED and no episode ever opened.
    assert seams.built_providers == ["cheap-1", "pricey-1"]
    assert seams.systems == []
    assert seams.world_model.tasks == []


def test_route_sweep_confirming_at_a_tty_runs_the_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seams = _patch_seams(monkeypatch)
    root = _project(tmp_path, traces=_corpus())
    monkeypatch.setattr(route_module, "_console", Console(force_terminal=True))
    monkeypatch.setattr(route_module, "Confirm", _Answer(True))
    out, result = _sweep(tmp_path, root, "support", "--scenarios", "1")
    assert result.exit_code == 0, result.output
    assert len(OutcomeMatrix.load(out).outcomes) == 2
    # Both candidates constructed twice: once by the pre-flight (before the cost question) and
    # once per cell by `evaluate_pool`, which still owns per-cell provider state.
    assert seams.built_providers == ["cheap-1", "pricey-1", "cheap-1", "pricey-1"]


def test_route_sweep_non_interactive_without_yes_says_it_could_not_ask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No TTY to prompt at and no --yes: every pool entry is priced, so the spend is accountable
    # and the run proceeds (the distill CLI's rule), but the log has to say that out loud.
    _patch_seams(monkeypatch)
    root = _project(tmp_path, traces=_corpus())
    out, result = _sweep(tmp_path, root, "support", "--scenarios", "1")
    assert result.exit_code == 0, result.output
    assert _says(result.output, "non-interactive session: proceeding without confirmation")
    assert len(OutcomeMatrix.load(out).outcomes) == 2


def test_route_sweep_cost_estimate_states_its_assumption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_seams(monkeypatch)
    root = _project(tmp_path, traces=_corpus())
    _out, result = _sweep(
        tmp_path,
        root,
        "support",
        "--scenarios",
        "3",
        "--max-steps",
        "20",
        "--assume-input-tokens",
        "2000",
        "--assume-output-tokens",
        "250",
        "--yes",
    )
    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    # 3 scenarios x 1 episode x 20 calls = 60 calls; cheap = 60 x (2000x1 + 250x2)/1e6 = $0.15,
    # pricey = 10x that, so the projected total is $1.65.
    assert "0.15" in flat and "1.50" in flat and "1.65" in flat
    assert "ASSUMED" in flat and "ASSUMPTION" in flat
    # The world model's own meter is excluded, and the table says so in plain words (no internal
    # decision id: "D12" means nothing to an operator reading a cost table).
    assert _says(result.output, "meteredseparatelyandareNOTinthisfigure")
    assert "D12" not in flat
    # ... and the measured spend is reported separately from the projection.
    assert "measuredcandidatespend" in flat


def test_route_sweep_rejects_a_missing_pool_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_seams(monkeypatch)
    root = _project(tmp_path, traces=_corpus())
    out = tmp_path / "matrix.json"
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "sweep",
            "support",
            "--root",
            str(root),
            "--pool",
            str(tmp_path / "nope.toml"),
            "--out",
            str(out),
            "--yes",
        ],
    )
    assert result.exit_code != 0
    flat = _flat(result.output)
    assert "nope.toml" in flat  # names the file it wanted
    assert "[[model]]" in flat and "input_per_mtok" in flat  # and how to write one
    assert not out.exists()


def test_route_sweep_rejects_a_missing_trace_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_seams(monkeypatch)
    root = _project(tmp_path, traces=None)  # a built model whose corpus was never kept
    out, result = _sweep(tmp_path, root, "support", "--yes")
    assert result.exit_code != 0
    flat = _flat(result.output)
    assert "notracecorpus" in flat
    assert flat.count(TRACES_FILENAME) >= 2  # both places it looked, so the fix is obvious
    assert not out.exists()


def test_route_sweep_rejects_an_unbuilt_world_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_seams(monkeypatch)
    root = _project(tmp_path, traces=_corpus())
    out, result = _sweep(tmp_path, root, "ghost", "--yes")
    assert result.exit_code != 0
    flat = _flat(result.output)
    assert "ghost" in flat and "wmobuild" in flat  # names the model and the command that fixes it
    assert not out.exists()


def test_route_sweep_says_when_a_tiny_corpus_is_not_leak_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Three train-band traces: the split leaves no held-out band, so the sweep falls back to the
    # whole corpus the way `wmo eval` does and must say the scenarios are not leak-free.
    _patch_seams(monkeypatch)
    root = _project(tmp_path, traces=_corpus(3))
    out, result = _sweep(tmp_path, root, "--scenarios", "2", "--yes")  # default model resolution
    assert result.exit_code == 0, result.output
    assert "notleak-free" in _flat(result.output)
    # Deterministic even here: the corpus is written newest-id-first, and `--scenarios` still cuts
    # the same by-trace-id prefix.
    assert OutcomeMatrix.load(out).scenario_ids() == ["tr-000", "tr-001"]


def test_route_sweep_takes_the_corpus_from_traces_for_a_locally_built_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `wmo build` keeps prompts, metrics and the index but NOT the corpus it read, so a model
    # built the canonical way has no sibling traces file. `--traces` is the only thing that makes
    # the documented call site work on such a project.
    _patch_seams(monkeypatch)
    root = _project(tmp_path, traces=None)  # built the canonical way: no corpus kept
    corpus_file = tmp_path / "elsewhere" / "traces.otel.jsonl"
    write_traces_jsonl(_corpus(), corpus_file)
    out, result = _sweep(
        tmp_path, root, "support", "--traces", str(corpus_file), "--scenarios", "2", "--yes"
    )
    assert result.exit_code == 0, result.output
    assert OutcomeMatrix.load(out).scenario_ids() == list(_HELD_OUT_IDS[:2])


def test_route_sweep_rejects_a_traces_file_that_is_not_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_seams(monkeypatch)
    root = _project(tmp_path, traces=None)
    out, result = _sweep(
        tmp_path, root, "support", "--traces", str(tmp_path / "gone.jsonl"), "--yes"
    )
    assert result.exit_code != 0
    flat = _flat(result.output)
    assert "gone.jsonl" in flat and "--traces" in flat
    assert not out.exists()


def test_route_sweep_names_the_positional_when_the_model_is_ambiguous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `WorldModelStore.resolve` says "pass --name", the option `wmo serve`/`play`/`demo` carry.
    # This command takes the model positionally, so following that advice fails; the message has
    # to name what a user of THIS command types.
    _patch_seams(monkeypatch)
    root = _project(tmp_path, traces=_corpus())
    _project(tmp_path, traces=_corpus())  # same root
    save_config(
        HarnessConfig(
            providers=[ProviderConfig(kind=ProviderKind.ANTHROPIC, model="fake-serve")],
            serve_provider=ProviderKind.ANTHROPIC,
        ),
        root / "models" / "other",
    )
    out, result = _sweep(tmp_path, root, "--yes")  # no MODEL, two models built
    assert result.exit_code != 0
    flat = _flat(result.output)
    assert "MODEL" in flat and "other,support" in flat
    assert "--name" not in flat
    assert "wmooptimizeroutesweepother" in flat  # a command that actually works
    assert not out.exists()


def test_route_sweep_checks_candidate_credentials_before_it_spends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `pool_provider` reads `api_key_env` per cell, so an unset variable on the SECOND candidate
    # used to abort mid-sweep with a raw ValueError after the first was fully paid for.
    seams = _patch_seams(monkeypatch)
    monkeypatch.delenv("WMO_TEST_MISSING_KEY", raising=False)
    root = _project(tmp_path, traces=_corpus())
    pool = tmp_path / "pool.toml"
    pool.write_text(
        "[[model]]\n"
        'name = "cheap"\n'
        'kind = "openai"\n'
        'model = "cheap-1"\n'
        "input_per_mtok = 1.0\n"
        "output_per_mtok = 2.0\n"
        "\n"
        "[[model]]\n"
        'name = "pricey"\n'
        'kind = "openai"\n'
        'model = "pricey-1"\n'
        'api_key_env = "WMO_TEST_MISSING_KEY"\n'
        "input_per_mtok = 10.0\n"
        "output_per_mtok = 20.0\n",
        encoding="utf-8",
    )
    out = tmp_path / "matrix.json"
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "sweep",
            "support",
            "--root",
            str(root),
            "--pool",
            str(pool),
            "--out",
            str(out),
            "--scenarios",
            "2",
            "--yes",
        ],
    )
    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit)  # no traceback
    flat = _flat(result.output)
    assert "WMO_TEST_MISSING_KEY" in flat and "pricey" in flat
    # Nothing was paid for: the check runs before the cost table, so no candidate was ever called
    # and no episode ever opened. (The pre-flight does construct the candidates it can, which is
    # free; `cheap` resolves, `pricey` never gets that far.)
    assert seams.built_providers == ["cheap-1"]
    assert seams.systems == []
    assert seams.world_model.tasks == []
    assert not out.exists()


def test_route_sweep_constructs_every_backend_before_it_spends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A credential check is not an availability check. `TinkerChatProvider` REFUSES an explicit
    # api_key (it authenticates through the shared service client), and nothing rejects that
    # combination at load, so the failure used to land at this candidate's FIRST CELL: after
    # `cheap` had run every scenario and been paid for, as a raw traceback, with no matrix.
    # `real_kinds` builds the real tinker backend here; `cheap` stays faked, and no candidate is
    # ever called, so the test makes no network request either way.
    seams = _patch_seams(monkeypatch, real_kinds=frozenset({ProviderKind.TINKER}))
    monkeypatch.setenv("WMO_TEST_TINKER_KEY", "sk-present")
    root = _project(tmp_path, traces=_corpus())
    pool = tmp_path / "pool.toml"
    pool.write_text(
        "[[model]]\n"
        'name = "cheap"\n'
        'kind = "openai"\n'
        'model = "cheap-1"\n'
        "input_per_mtok = 1.0\n"
        "output_per_mtok = 2.0\n"
        "\n"
        "[[model]]\n"
        'name = "student"\n'
        'kind = "tinker"\n'
        'model = "Qwen/Qwen3-8B"\n'
        'api_key_env = "WMO_TEST_TINKER_KEY"\n'  # set, so this is NOT a credential failure
        "input_per_mtok = 0.1\n"
        "output_per_mtok = 0.2\n",
        encoding="utf-8",
    )
    out = tmp_path / "matrix.json"
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "sweep",
            "support",
            "--root",
            str(root),
            "--pool",
            str(pool),
            "--out",
            str(out),
            "--scenarios",
            "2",
            "--yes",
        ],
    )
    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit)  # no traceback
    flat = _flat(result.output)
    # Names the offending entry AND its kind, so the pool file is editable from the message, plus
    # what to do about it (the backend's own advice: drop api_key_env, export TINKER_API_KEY).
    assert "'student'" in flat and "kind=tinker" in flat
    assert "TINKER_API_KEY" in flat and "dropapi_key_env" in flat
    # Not one cell was paid for: the cost table never printed and no episode ever opened.
    assert "USD(est)" not in flat
    assert seams.systems == []
    assert seams.world_model.tasks == []
    assert not out.exists()


def test_route_sweep_rejects_a_config_no_backend_could_use_before_it_spends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The STATIC half of the pre-flight. An azure entry with a deployment but no `api_version`
    # loads (the load-time rule only covers `deployment`) and CONSTRUCTS (the provider's __init__
    # just stores the config); only `_get_client` refuses without an api-version, and that runs
    # inside this candidate's first call, after `cheap` has been paid for. Nothing about this needs
    # an SDK or a credential, so it is knowable from the entry alone.
    seams = _patch_seams(monkeypatch)
    root = _project(tmp_path, traces=_corpus())
    pool = tmp_path / "pool.toml"
    pool.write_text(
        "[[model]]\n"
        'name = "cheap"\n'
        'kind = "openai"\n'
        'model = "cheap-1"\n'
        "input_per_mtok = 1.0\n"
        "output_per_mtok = 2.0\n"
        "\n"
        "[[model]]\n"
        'name = "gpt-azure"\n'
        'kind = "azure"\n'
        'model = "gpt-5.5"\n'
        'deployment = "gpt-5.5"\n'  # present, so this is not the load-time rule
        'endpoint = "https://example.openai.azure.com"\n'
        "input_per_mtok = 1.0\n"
        "output_per_mtok = 2.0\n",
        encoding="utf-8",
    )
    out = tmp_path / "matrix.json"
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "sweep",
            "support",
            "--root",
            str(root),
            "--pool",
            str(pool),
            "--out",
            str(out),
            "--scenarios",
            "2",
            "--yes",
        ],
    )
    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit)  # no traceback
    flat = _flat(result.output)
    # Names the offending entry, its kind, and the field to add.
    assert "'gpt-azure'" in flat and "kind=azure" in flat and "api_version" in flat
    # Before the cost confirmation, and before any cell: the cost table never printed, no candidate
    # was ever called, no episode was ever opened, and no matrix exists.
    assert "USD(est)" not in flat
    assert seams.systems == []
    assert seams.world_model.tasks == []
    assert not out.exists()


def test_route_sweep_builds_every_lazy_client_before_it_spends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The LAZY-CLIENT half of the pre-flight. Constructing an `OpenAIProvider` does not read its
    # credential: `__init__` only stores the config, and `OpenAI()` (which REFUSES to construct
    # without a resolvable key) is built inside the first call. So with OPENAI_API_KEY unset, a
    # pre-flight that only constructs providers passes and the whole sweep then fails cell by cell.
    # Both candidates are the real backend here; no request is made either way, because the SDK
    # raises while building its own client.
    seams = _patch_seams(monkeypatch, real_kinds=frozenset({ProviderKind.OPENAI}))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    root = _project(tmp_path, traces=_corpus())
    out, result = _sweep(tmp_path, root, "support", "--scenarios", "2", "--yes")
    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit)  # no traceback
    flat = _flat(result.output)
    # EVERY unusable candidate is named with its kind, not just the first one an operator would fix.
    assert "'cheap'" in flat and "'pricey'" in flat and "kind=openai" in flat
    assert "OPENAI_API_KEY" in flat  # the SDK's own advice survives into the message
    assert "USD(est)" not in flat
    assert seams.world_model.tasks == []
    assert not out.exists()


def test_route_sweep_rejects_a_bedrock_candidate_whose_region_resolves_nowhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bedrock is the backend whose client CANNOT be built in a pre-flight (boto3 resolves
    # credentials by walking a chain that reaches the instance-metadata endpoint, and builds fine
    # with no credentials anyway), so its region is resolved through boto3's own session instead:
    # entry, then AWS_DEFAULT_REGION, then the active profile. Without that check, botocore's
    # NoRegionError lands in this candidate's first cell. Every source is pointed at nothing here so
    # the check has the same answer on any machine, and metadata lookups are disabled as a belt:
    # this test may not touch the network.
    seams = _patch_seams(monkeypatch, real_kinds=frozenset({ProviderKind.BEDROCK}))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-aws-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-aws-credentials"))
    for name in ("AWS_REGION", "AWS_DEFAULT_REGION", "AWS_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    root = _project(tmp_path, traces=_corpus())
    pool = tmp_path / "pool.toml"
    pool.write_text(
        "[[model]]\n"
        'name = "cheap"\n'
        'kind = "openai"\n'
        'model = "cheap-1"\n'
        "input_per_mtok = 1.0\n"
        "output_per_mtok = 2.0\n"
        "\n"
        "[[model]]\n"
        'name = "opus-bedrock"\n'
        'kind = "bedrock"\n'
        'model = "us.anthropic.claude-opus-4-8"\n'  # no region anywhere
        "input_per_mtok = 15.0\n"
        "output_per_mtok = 75.0\n",
        encoding="utf-8",
    )
    out = tmp_path / "matrix.json"
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "sweep",
            "support",
            "--root",
            str(root),
            "--pool",
            str(pool),
            "--out",
            str(out),
            "--scenarios",
            "2",
            "--yes",
        ],
    )
    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit)  # no traceback
    flat = _flat(result.output)
    assert "'opus-bedrock'" in flat and "kind=bedrock" in flat
    assert "region" in flat and "AWS_DEFAULT_REGION" in flat  # what went wrong and what to do
    assert "USD(est)" not in flat
    assert seams.systems == []
    assert seams.world_model.tasks == []
    assert not out.exists()


def test_route_sweep_states_what_the_preflight_cannot_know_before_the_first_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two backends keep a residual gap because closing it needs a request (bedrock AWS credentials,
    # tinker service reachability). A usable bedrock entry therefore passes the pre-flight and the
    # sweep runs, and the command says which entry still carries which unknown rather than leaving
    # an operator to find out mid-sweep. Faked provider construction: nothing is called for real.
    seams = _patch_seams(monkeypatch)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")  # so the region check passes
    root = _project(tmp_path, traces=_corpus())
    pool = tmp_path / "pool.toml"
    pool.write_text(
        "[[model]]\n"
        'name = "opus-bedrock"\n'
        'kind = "bedrock"\n'
        'model = "us.anthropic.claude-opus-4-8"\n'
        "input_per_mtok = 15.0\n"
        "output_per_mtok = 75.0\n",
        encoding="utf-8",
    )
    out = tmp_path / "matrix.json"
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "sweep",
            "support",
            "--root",
            str(root),
            "--pool",
            str(pool),
            "--out",
            str(out),
            "--scenarios",
            "1",
            "--yes",
        ],
    )
    assert result.exit_code == 0, result.output
    assert _says(result.output, "opus-bedrock (kind=bedrock): AWS credentials")
    assert _says(result.output, "the pre-flight makes no request")
    # ... and WHEN it would land, so the note is a bound rather than an open-ended caveat.
    assert _says(result.output, "lands in the first 1 cell(s) of the sweep")
    assert seams.world_model.tasks == ["task tr-010"]  # the sweep did run


def test_route_sweep_checks_the_out_path_before_it_spends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `OutcomeMatrix.save` mkdirs the parent at the END of the sweep, so a parent component that
    # is a regular file used to discard every cell already paid for with a bare OS error.
    seams = _patch_seams(monkeypatch)
    root = _project(tmp_path, traces=_corpus())
    blocker = tmp_path / "blocker"
    blocker.write_text("a regular file, not a directory", encoding="utf-8")
    out = blocker / "matrix.json"
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "sweep",
            "support",
            "--root",
            str(root),
            "--pool",
            str(_pool_file(tmp_path)),
            "--out",
            str(out),
            "--scenarios",
            "2",
            "--yes",
        ],
    )
    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit)  # no traceback
    assert _says(result.output, "cannot write the outcome matrix")
    assert seams.world_model.tasks == []  # no episode ever opened
    # The check is pure: an --out it refuses is not half-created on the way out.
    assert blocker.is_file() and not out.exists()


def test_route_sweep_prints_names_and_paths_rich_cannot_swallow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pool names are free-form operator strings and --out is a path: both reach a rich console,
    # where `[a]` is markup. Unescaped, the cost table showed two candidates as one name and the
    # handoff line printed a path that does not exist.
    _patch_seams(monkeypatch)
    root = _project(tmp_path, traces=_corpus())
    pool = tmp_path / "pool.toml"
    pool.write_text(
        "[[model]]\n"
        'name = "gpt[a]"\n'
        'kind = "openai"\n'
        'model = "cheap-1"\n'
        "input_per_mtok = 1.0\n"
        "output_per_mtok = 2.0\n"
        "\n"
        "[[model]]\n"
        'name = "gpt[/bold]"\n'
        'kind = "openai"\n'
        'model = "pricey-1"\n'
        "input_per_mtok = 10.0\n"
        "output_per_mtok = 20.0\n",
        encoding="utf-8",
    )
    out = tmp_path / "[run1]" / "matrix.json"
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "sweep",
            "support",
            "--root",
            str(root),
            "--pool",
            str(pool),
            "--out",
            str(out),
            "--scenarios",
            "1",
            "--yes",
        ],
    )
    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    # Both candidates are distinguishable in the table the operator confirms spend from, and the
    # closing-tag name no longer aborts the command before any episode runs.
    assert "gpt[a]" in flat and "gpt[/bold]" in flat
    # The printed path is the path the matrix is actually at, so it can be copied.
    assert out.is_file()
    assert _says(result.output, str(out))
    assert _says(result.output, f"wmo optimize route fit {out} --kind knn")
