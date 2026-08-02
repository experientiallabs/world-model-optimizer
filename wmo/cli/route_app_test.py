"""CLI tests for `wmo optimize route` (fit + report), driven via CliRunner."""

from __future__ import annotations

import importlib
import itertools
import json
import sys
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import pytest
from filelock import FileLock
from rich.console import Console
from typer.testing import CliRunner, Result

import wmo.env as env_module
from wmo.cli import consent as consent_module
from wmo.cli.app import app
from wmo.config import HarnessConfig, save_config
from wmo.core.types import Action, ActionKind, EnvState, Observation, Session, Step, Trace
from wmo.distill.store import DistillModelCard
from wmo.engine.world_model import WorldModel
from wmo.env.llm_agent import DEFAULT_HISTORY_CHARS
from wmo.ingest.otel_writer import write_traces_jsonl
from wmo.optimize.compression import (
    CompressionConfig,
    Compressor,
    TruncateCompressor,
    register_compressor,
    registered_compressor_ids,
    same_compression,
)
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.policy import POLICY_FILENAME, RoutingPolicy, select_model
from wmo.optimize.reward import EpisodeScore
from wmo.optimize.routing import evaluate_policy
from wmo.optimize.sweep_partial import PartialHeader, PlanIdentity
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
from wmo.providers.openrouter import OPENROUTER_API_KEY_ENV
from wmo.providers.pool import PoolEntry, load_pool
from wmo.providers.registry import get_provider as registry_get_provider
from wmo.serving.traces_source import TRACES_FILENAME
from wmo.tracking import Phase, RunRecord, UsageTotals, load_runs

runner = CliRunner()


def _arm(compression: CompressionConfig | None) -> tuple[str, str, float]:
    """The D-COMPRESS fields an episode measured under `compression` would carry.

    A matrix records the arm its rewards were produced under, and `fit` refuses to stamp a
    policy whose compression config disagrees, so a fixture fitting `--compressor` has to look
    like episodes that actually ran that way.
    """
    if compression is None:
        return "", "", 0.0
    return (
        compression.compressor_id,
        compression.compressor_version,
        compression.aggressiveness,
    )


def _matrix_file(tmp_path: Path, *, compression: CompressionConfig | None = None) -> Path:
    """The uncompressed arm by default; `compression` stamps the rows as that arm instead."""
    pool = [
        PoolEntry(
            name="a", kind=ProviderKind.OPENAI, model="a", input_per_mtok=1.0, output_per_mtok=1.0
        ),
        PoolEntry(
            name="b", kind=ProviderKind.OPENAI, model="b", input_per_mtok=1.0, output_per_mtok=1.0
        ),
    ]
    arm_id, arm_version, arm_aggressiveness = _arm(compression)
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
                    compressor_id=arm_id,
                    compressor_version=arm_version,
                    aggressiveness=arm_aggressiveness,
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
            "--kind",
            "rank",
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
    assert set(policy.fit_scenario_ids).isdisjoint(report["scenario_ids"])
    assert len(policy.fit_scenario_ids) + report["scenario_count"] == 4
    assert report["cost_assumptions"]


def _fit_then_report(tmp_path: Path, *extra: str) -> tuple[Result, Path]:
    """Fit a knn policy, then report over the same matrix with `extra` report flags.

    knn because only a dialable policy produces routed detents, and so a curve at all.
    """
    matrix_file = _knn_matrix_file(tmp_path)
    policy_file = tmp_path / "policy.json"
    fit = runner.invoke(
        app,
        [
            *("optimize", "route", "fit", str(matrix_file)),
            *("--kind", "knn", "--fallback", "a", "--out", str(policy_file)),
            *("--z", "0.5", "--rag-num", "3", "--min-pairs", "2"),
        ],
    )
    assert fit.exit_code == 0, fit.output
    report_file = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            *("optimize", "route", "report", str(matrix_file), str(policy_file)),
            *("--baseline", "a", "--out", str(report_file)),
            *extra,
        ],
    )
    return result, report_file.parent / "pareto.json"


def test_the_curve_defaults_to_the_world_model_labels(tmp_path: Path) -> None:
    # Unchanged behavior for every existing caller: a sweep's matrix was scored by the world
    # model's own verifier.
    result, pareto = _fit_then_report(tmp_path)
    assert result.exit_code == 0, result.output
    curve = json.loads(pareto.read_text())
    assert curve["provenance"] == "wm_simulated"
    assert curve["judge"] == "world-model verifier"


def test_a_real_benchmark_matrix_can_label_its_own_curve(tmp_path: Path) -> None:
    # The bench-defaults case: the rewards are real tau2 episodes, and a curve claiming they came
    # out of a world model would present a measurement as a simulation. ParetoCurve.provenance
    # exists to stop exactly that, and until now the CLI hardcoded it.
    result, pareto = _fit_then_report(
        tmp_path, "--provenance", "real_episode", "--judge", "tau2 reward"
    )
    assert result.exit_code == 0, result.output
    curve = json.loads(pareto.read_text())
    assert curve["provenance"] == "real_episode"
    assert curve["judge"] == "tau2 reward"


def test_a_misspelled_provenance_is_refused_not_written(tmp_path: Path) -> None:
    result, pareto = _fit_then_report(tmp_path, "--provenance", "real")
    assert result.exit_code != 0
    assert "real_episode" in result.output
    assert not pareto.exists()


def test_the_report_label_defaults_to_the_world_model_phrasing(tmp_path: Path) -> None:
    result, pareto = _fit_then_report(tmp_path)
    assert result.exit_code == 0, result.output
    report = json.loads((pareto.parent / "report.json").read_text())
    assert "reconstructed from your traces" in report["scenario_label"]


def test_a_real_benchmark_report_can_say_what_it_measured(tmp_path: Path) -> None:
    # scenario_label is the one line of the report a customer actually reads. Telling them their
    # endpoint was measured on scenarios "reconstructed from your traces" when it was measured on
    # a pinned public benchmark is false, and until now the phrasing was hardcoded.
    result, pareto = _fit_then_report(
        tmp_path, "--scenario-label", "on the 20 pinned tau2-bench eval tasks"
    )
    assert result.exit_code == 0, result.output
    report = json.loads((pareto.parent / "report.json").read_text())
    assert report["scenario_label"] == "on the 20 pinned tau2-bench eval tasks"


def test_route_fit_rejects_unknown_embedder(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["optimize", "route", "fit", str(_matrix_file(tmp_path)), "--embedder", "vibes"],
    )
    assert result.exit_code != 0
    assert "hashing, openai or azure" in result.output


def _knn_matrix_file(
    tmp_path: Path,
    *,
    flip: bool = False,
    name: str = "knn_matrix.json",
    compression: CompressionConfig | None = None,
) -> Path:
    """Twelve scenarios: enough neighbors per query for a guarded fit to route at all.

    `flip` swaps which model wins each half, so two matrices built here disagree on every cell.
    That is what makes an artifact mix-up observable: a policy fitted on one and served the
    other's evidence routes every request the wrong way.
    """
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
    arm_id, arm_version, arm_aggressiveness = _arm(compression)
    outcomes = []
    for group, tasks in (("sql", sql), ("prose", prose)):
        for index, task in enumerate(tasks):
            for model in ("a", "b"):
                wins = ((model == "a") == (group == "sql")) != flip
                outcomes.append(
                    ScenarioOutcome(
                        scenario_id=f"{group}:{index}",
                        task=task,
                        model=model,
                        reward=1.0 if wins else 0.0,
                        success=wins,
                        cost_usd=0.001,
                        compressor_id=arm_id,
                        compressor_version=arm_version,
                        aggressiveness=arm_aggressiveness,
                    )
                )
    path = tmp_path / name
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
    # The sidecar is named after --out and recorded in the policy, not resolved by convention.
    assert policy.knn_bank_path == "policy.json.bank.npz"
    assert policy.bank_path() == tmp_path / "policy.json.bank.npz"
    assert policy.bank_path().is_file()  # sidecar beside the policy
    assert len(policy.knn_bank().scenario_ids) == 8
    assert policy.fit_scenario_ids == policy.knn_bank().scenario_ids
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


def _fit_knn(matrix_file: Path, policy_file: Path, *, fallback: str = "a") -> Result:
    """Run `route fit --kind knn` with the neighbor budget this twelve-scenario matrix needs."""
    return runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(matrix_file),
            "--kind",
            "knn",
            "--fallback",
            fallback,
            "--rag-num",
            "3",
            "--min-pairs",
            "2",
            "--out",
            str(policy_file),
        ],
    )


def _fitted_knn_policy(tmp_path: Path) -> Path:
    policy_file = tmp_path / POLICY_FILENAME
    result = _fit_knn(_knn_matrix_file(tmp_path), policy_file)
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
        [
            "optimize",
            "route",
            "fit",
            str(_matrix_file(tmp_path)),
            "--kind",
            "rank",
            "--out",
            str(policy_file),
        ],
    )
    assert fit.exit_code == 0, fit.output
    result = runner.invoke(
        app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "0.5"]
    )
    assert result.exit_code != 0
    assert "kind='rank'" in result.output


def test_route_fit_knn_gives_each_policy_its_own_evidence_bank(tmp_path: Path) -> None:
    """Two knn fits into one directory must not share (and overwrite) one sidecar.

    Regression: the bank name used to be hard-coded, so the second fit clobbered the first
    policy's evidence and both policies recorded the same relative path. Policy A then served
    matrix B's rewards, which inverts every routing decision on this pair of matrices.
    """
    a_matrix = _knn_matrix_file(tmp_path, name="matrix_a.json")
    b_matrix = _knn_matrix_file(tmp_path, flip=True, name="matrix_b.json")
    # The third fit shares a STEM with the first: the bank name is appended to the policy
    # filename rather than substituted for its extension, so it still gets its own sidecar.
    fits = (
        ("policy_a.json", a_matrix, "a"),
        ("policy_b.json", b_matrix, "b"),
        ("policy_a.yaml", b_matrix, "b"),
    )
    for name, matrix_file, fallback in fits:
        result = _fit_knn(matrix_file, tmp_path / name, fallback=fallback)
        assert result.exit_code == 0, result.output

    policies = [RoutingPolicy.load(tmp_path / name) for name, _, _ in fits]
    assert [policy.knn_bank_path for policy in policies] == [
        "policy_a.json.bank.npz",
        "policy_b.json.bank.npz",
        "policy_a.yaml.bank.npz",
    ]
    banks = [policy.bank_path() for policy in policies]
    assert len(set(banks)) == len(banks)
    assert all(bank.is_file() for bank in banks)
    # Policy A still routes on ITS evidence: prose is b's half of matrix_a, and matrix_b says
    # the opposite, so this is 1.0 only if the later fits left A's bank alone.
    matrix = OutcomeMatrix.load(a_matrix)
    prose_ids = [sid for sid in matrix.scenario_ids() if sid.startswith("prose:")]
    assert evaluate_policy(policies[0], matrix, prose_ids).model_mix == {"b": 1.0}


def test_route_tune_refuses_a_base_snapshot_from_a_superseded_fit(tmp_path: Path) -> None:
    """fit -> tune -> refit -> tune must not silently dial the pre-refit artifact.

    Regression: `tune` always re-read `<stem>.base.json`, which `fit` never invalidates, so the
    second tune reported success while overwriting the new fit with a dialed copy of the old one.
    """
    policy_file = _fitted_knn_policy(tmp_path)
    tuned = runner.invoke(
        app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "0.6"]
    )
    assert tuned.exit_code == 0, tuned.output

    refit = _fit_knn(
        _knn_matrix_file(tmp_path, flip=True, name="refit_matrix.json"),
        policy_file,
        fallback="b",
    )
    assert refit.exit_code == 0, refit.output
    assert RoutingPolicy.load(policy_file).default_model == "b"

    stale = runner.invoke(
        app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "0.3"]
    )
    assert stale.exit_code != 0
    assert _says(stale.output, "as-fitted snapshot of a different fit")
    assert _says(stale.output, "policy.base.json")  # names the file to delete
    # The refit survives untouched rather than being replaced by a dialed copy of the old fit.
    after = RoutingPolicy.load(policy_file)
    assert after.default_model == "b"
    assert after.cost_quality is None


def test_route_fit_digests_the_bytes_it_actually_fitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recorded digest must describe the matrix the fit SAW, not a later read of the path.

    Regression: `fit` parsed the matrix and then re-read the file to digest it. A corpus rebuilt
    in place between the two stamped the old fit with the new file's digest, so the next fit of
    that new matrix matched its provenance and `tune` accepted the superseded snapshot -- the
    very failure the digest exists to catch. The two now come out of one read.
    """
    matrix_file = _knn_matrix_file(tmp_path)
    fitted_bytes = matrix_file.read_bytes()
    real = {name: getattr(Path, name) for name in ("read_bytes", "read_text")}

    def _rebuilding(name: str):  # noqa: ANN202 - the wrapped reader's own signature
        def _read(self: Path, *args: object, **kwargs: object):  # noqa: ANN202
            payload = real[name](self, *args, **kwargs)
            if self == matrix_file:  # swap the corpus the instant the fit has read it
                monkeypatch.undo()  # ...once, whichever reader the fit happens to use
                _knn_matrix_file(tmp_path, flip=True)
            return payload

        return _read

    for name in real:
        monkeypatch.setattr(Path, name, _rebuilding(name))
    policy_file = tmp_path / POLICY_FILENAME
    assert _fit_knn(matrix_file, policy_file).exit_code == 0
    assert matrix_file.read_bytes() != fitted_bytes  # the file on disk did change under the fit

    # The digest is of the bytes that were fitted, so a later fit of the REPLACEMENT differs.
    fitted_from = RoutingPolicy.load(policy_file).fitted_from or ""
    other = tmp_path / "other.json"
    assert _fit_knn(matrix_file, other).exit_code == 0
    assert fitted_from != (RoutingPolicy.load(other).fitted_from or "")


def test_route_tune_refuses_a_snapshot_after_the_matrix_was_rebuilt_in_place(
    tmp_path: Path,
) -> None:
    """Same matrix path, same flags, different contents: still a different fit.

    A corpus is routinely rebuilt under the filename it already had, so a path alone cannot
    identify a fit. `fitted_from` carries a digest of the matrix, which is what makes the
    snapshot check catch this.
    """
    policy_file = _fitted_knn_policy(tmp_path)
    tuned = runner.invoke(
        app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "0.6"]
    )
    assert tuned.exit_code == 0, tuned.output

    rebuilt = _knn_matrix_file(tmp_path, flip=True)  # same default filename, opposite labels
    refit = _fit_knn(rebuilt, policy_file)  # and the same fit flags as `_fitted_knn_policy`
    assert refit.exit_code == 0, refit.output

    stale = runner.invoke(
        app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "0.3"]
    )
    assert stale.exit_code != 0
    assert _says(stale.output, "as-fitted snapshot of a different fit")
    assert RoutingPolicy.load(policy_file).cost_quality is None  # the refit is untouched


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows rejects '|' in path components used by this regression fixture",
)
def test_route_tune_survives_a_matrix_path_that_looks_like_a_dial_suffix(tmp_path: Path) -> None:
    """An operator-supplied path must not be able to truncate the fit identity it opens.

    Regression: `fit_provenance` split on the FIRST ` | cost_quality=`, and `fitted_from` starts
    with the matrix path. A path containing that substring therefore discarded the digest and
    every fit flag behind it, collapsing unrelated fits onto one identity, and `tune` dialed the
    superseded snapshot over the refit. This name carries a COMPLETE, well-formed dial suffix,
    which is the worst case: the fit flags follow the path, so the real suffix is still the only
    one at the end of the string.
    """
    hostile = "m | cost_quality=0.5 (floor_q=0.05, lam=0, guard=symmetric).json"
    policy_file = tmp_path / POLICY_FILENAME
    assert _fit_knn(_knn_matrix_file(tmp_path, name=hostile), policy_file).exit_code == 0
    for dial in ("0.6", "0.2"):  # no refit between these: the dial must still move freely
        tuned = runner.invoke(
            app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", dial]
        )
        assert tuned.exit_code == 0, tuned.output
    assert RoutingPolicy.load(policy_file).cost_quality == 0.2

    rebuilt = _knn_matrix_file(tmp_path, flip=True, name=hostile)  # same path, opposite labels
    assert _fit_knn(rebuilt, policy_file, fallback="b").exit_code == 0
    stale = runner.invoke(
        app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "0.3"]
    )
    assert stale.exit_code != 0
    assert _says(stale.output, "as-fitted snapshot of a different fit")
    after = RoutingPolicy.load(policy_file)
    assert after.default_model == "b"  # the refit survives, not a dialed copy of the old fit
    assert after.cost_quality is None


def test_route_tune_that_fails_leaves_no_base_snapshot_behind(tmp_path: Path) -> None:
    """A rejected tune must not poison the path for the next fit.

    Regression: the base snapshot was copied before validation, so a failed tune left a stray
    `policy.base.json`. A later `fit --kind knn` into the same path could never be tuned: the
    error reported kind='rank' while the policy on disk was demonstrably kind='knn'.
    """
    policy_file = tmp_path / POLICY_FILENAME
    fit = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(_matrix_file(tmp_path)),
            "--kind",
            "rank",
            "--out",
            str(policy_file),
        ],
    )
    assert fit.exit_code == 0, fit.output
    rejected = runner.invoke(
        app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "0.5"]
    )
    assert rejected.exit_code != 0
    assert "kind='rank'" in rejected.output
    assert not (tmp_path / "policy.base.json").exists()

    # The path is still tunable once a knn policy is fitted into it.
    refit = _fit_knn(_knn_matrix_file(tmp_path), policy_file)
    assert refit.exit_code == 0, refit.output
    tuned = runner.invoke(
        app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "0.5"]
    )
    assert tuned.exit_code == 0, tuned.output
    assert RoutingPolicy.load(policy_file).cost_quality == 0.5


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


def test_route_pin_warns_when_out_bypasses_the_model_dir(tmp_path: Path) -> None:
    """A scratch --out succeeds but serving never sees it; the pin must say so.

    Both bench-defaults lanes shipped an endpoint whose model dir still held
    the OLD policy because `pin --out /tmp/...` printed the same success line
    as an in-place pin (2026-07-29).
    """
    pool_file = tmp_path / "pool.toml"
    assert _add_student(tmp_path, pool_file).exit_code == 0
    _built_model(tmp_path)
    scratch = tmp_path / "scratch" / "policy-pin.json"
    scratch.parent.mkdir()

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
            "--out",
            str(scratch),
        ],
    )

    assert result.exit_code == 0, result.output
    assert scratch.is_file()  # the pin still lands where asked
    assert "does NOT update" in result.output  # but the operator is told serving will not see it


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
    pool_file.parent.mkdir(parents=True, exist_ok=True)
    holder = FileLock(pool_file.with_name(f"{pool_file.name}.lock"))
    holder.acquire()
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
        holder.release()

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

    def __init__(
        self,
        reward: float = 0.75,
        judge_fails_on: frozenset[str] = frozenset(),
        session_usd: float = 0.0,
    ) -> None:
        self._reward = reward
        # What this fake charges per session for its OWN serve + judge calls, which is the
        # world-model side of a sweep's bill. Zero by default so existing expectations are
        # unchanged; a test that cares about that side sets it.
        self._session_usd = session_usd
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
        return self._usage_record(session_id)

    def session_usage(self, session_id: str) -> RunRecord:
        return self._usage_record(session_id)

    def _usage_record(self, session_id: str) -> RunRecord:
        totals = UsageTotals(
            calls=1, input_tokens=100, output_tokens=20, cost_usd=self._session_usd
        )
        return RunRecord(
            run_id=session_id,
            kind="serve",
            duration_seconds=0.5,
            total=totals,
            by_phase={Phase.SERVE: totals},
        )


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
    session_usd: float = 0.0,
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
            scenario. `evaluate_pool` builds one provider per episode in scenario-major order, so
            the cycle position is the episode index within the scenario. This is how a candidate
            comes back with the same scenarios as the others but FEWER scored episodes on them.
        judge_fails_on: Scenario tasks whose episode SCORING raises, for every candidate: the
            whole pool loses those scenarios together.
        real_kinds: Provider kinds to construct for real instead of faking, so a test can exercise
            a backend that refuses its own config or cannot build its lazy client. Construction and
            preparation are both request-free, and no real provider is ever called (the sweep must
            fail before any cell runs).
    """
    seams = _Seams(
        _FakeWorldModel(reward=reward, judge_fails_on=judge_fails_on, session_usd=session_usd)
    )
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

    monkeypatch.setattr("wmo.engine.load_world_model", _load)
    monkeypatch.setattr("wmo.providers.pool.get_provider", _get_provider)
    if no_scoring:
        real = env_module.WorldModelEnv
        monkeypatch.setattr(
            env_module,
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
            "--kind",
            "rank",
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


def test_route_sweep_at_higher_concurrency_writes_the_same_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `--concurrency` is a speed knob the operator can see in the plan, not a change of evidence:
    # the same cells, the same rows, in the same order.
    _patch_seams(monkeypatch)
    root = _project(tmp_path, traces=_corpus())
    out, result = _sweep(
        tmp_path, root, "support", "--scenarios", "3", "--concurrency", "4", "--yes"
    )
    assert result.exit_code == 0, result.output
    assert _says(result.output, "4 cell(s) run at once")
    matrix = OutcomeMatrix.load(out)
    assert [(o.model, o.scenario_id) for o in matrix.outcomes] == [
        (model, sid) for model in ("cheap", "pricey") for sid in _HELD_OUT_IDS[:3]
    ]
    assert all(o.scored for o in matrix.outcomes)


def test_route_sweep_resumes_the_cells_an_interrupted_run_already_bought(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A grid killed mid-flight is finished, not repeated: the filed gap, end to end.

    The first attempt dies inside its fourth cell. The rows it completed are on disk beside the
    matrix, so the second attempt measures only what is missing and the matrix it writes is the
    whole grid.
    """
    seams = _patch_seams(monkeypatch)
    root = _project(tmp_path, traces=_corpus())
    real_env = env_module.WorldModelEnv
    cells = itertools.count(1)

    class _DiesOnTheFourthCell:
        def __init__(self, world_model: object, *, score_on_close: bool = False) -> None:
            self._inner = real_env(cast("WorldModel", world_model), score_on_close=score_on_close)
            self._n = next(cells)

        def reset(self, task: str | None = None, seed_state: EnvState | None = None) -> EnvState:
            if self._n == 4:
                raise RuntimeError("simulated transport fault")
            return self._inner.reset(task=task, seed_state=seed_state)

        def __getattr__(self, name: str) -> object:
            return getattr(self._inner, name)

    monkeypatch.setattr(env_module, "WorldModelEnv", _DiesOnTheFourthCell)
    out, first = _sweep(tmp_path, root, "support", "--scenarios", "3", "--yes")
    assert first.exit_code != 0
    assert not out.exists()  # no matrix: the sweep never finished
    sidecar = out.with_name(out.name + ".partial.jsonl")
    assert len(sidecar.read_text(encoding="utf-8").splitlines()) == 4  # header + 3 paid cells

    monkeypatch.setattr(env_module, "WorldModelEnv", real_env)
    scored_before = len(seams.world_model.scored)
    _, second = _sweep(tmp_path, root, "support", "--scenarios", "3", "--yes")
    assert second.exit_code == 0, second.output
    assert _says(second.output, "RESUMING: 3 of those cell(s) are already measured")
    assert len(seams.world_model.scored) - scored_before == 3  # only the missing cells ran
    assert len(OutcomeMatrix.load(out).outcomes) == 6
    assert not sidecar.exists()


def test_route_sweep_refuses_a_sidecar_that_belongs_to_a_different_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Refused BEFORE the spend question, naming the pin that moved: two arms in one matrix is a
    # comparison nobody measured.
    _patch_seams(monkeypatch)
    root = _project(tmp_path, traces=_corpus())
    out, _first = _sweep(tmp_path, root, "support", "--scenarios", "3", "--yes")
    sidecar = out.with_name(out.name + ".partial.jsonl")
    # Re-create the sidecar a killed run would have left, then change what the sweep measures.
    sidecar.write_text(
        "\n".join(
            [
                PartialHeader(
                    identity=PlanIdentity(
                        pool="stale",
                        scenarios=tuple(_HELD_OUT_IDS[:3]),
                        episodes=1,
                        max_steps=20,
                        history_chars=DEFAULT_HISTORY_CHARS,
                        compression="raw text (no compression)",
                    )
                ).model_dump_json(),
                *(row.model_dump_json() for row in OutcomeMatrix.load(out).outcomes[:2]),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _, blocked = _sweep(tmp_path, root, "support", "--scenarios", "3", "--yes")
    assert blocked.exit_code != 0
    assert _says(blocked.output, "measured under a DIFFERENT plan")
    assert _says(blocked.output, "candidate pool changed")


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interactive_stdin: None
) -> None:
    seams = _patch_seams(monkeypatch)
    root = _project(tmp_path, traces=_corpus())
    monkeypatch.setattr(route_module, "_console", Console(force_terminal=True))
    monkeypatch.setattr(consent_module, "Confirm", _Answer(False))
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interactive_stdin: None
) -> None:
    seams = _patch_seams(monkeypatch)
    root = _project(tmp_path, traces=_corpus())
    monkeypatch.setattr(route_module, "_console", Console(force_terminal=True))
    monkeypatch.setattr(consent_module, "Confirm", _Answer(True))
    out, result = _sweep(tmp_path, root, "support", "--scenarios", "1")
    assert result.exit_code == 0, result.output
    assert len(OutcomeMatrix.load(out).outcomes) == 2
    # Both candidates constructed twice: once by the pre-flight (before the cost question) and
    # once per cell by `evaluate_pool`, which still owns per-cell provider state.
    assert seams.built_providers == ["cheap-1", "pricey-1", "cheap-1", "pricey-1"]


def test_route_sweep_non_interactive_without_yes_refuses_to_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No TTY to prompt at and no --yes: consent is said, never inferred. This branch used to
    # proceed-and-note; the equivalent branch in `optimize model` spent a scripted caller's
    # real money, so every spend surface now refuses (exit 2) and names the fix.
    _patch_seams(monkeypatch)
    root = _project(tmp_path, traces=_corpus())
    out, result = _sweep(tmp_path, root, "support", "--scenarios", "1")
    assert result.exit_code == 2, result.output
    assert _says(result.output, "cannot ask for spend consent")
    assert not Path(out).exists()  # nothing bought


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


def test_route_sweep_rejects_an_openrouter_candidate_with_no_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # OpenRouter joined the pre-flight late: it shipped with no `prepare` seam, so
    # `prepare_pool_provider` skipped it and an unset key landed at that candidate's FIRST CELL,
    # after `cheap` had run every scenario and been paid for. Unlike bedrock's credential this one
    # IS locally knowable: `OpenRouterProvider._get_client` resolves the key itself and refuses,
    # opening no connection, so no request is made either way.
    seams = _patch_seams(monkeypatch, real_kinds=frozenset({ProviderKind.OPENROUTER}))
    monkeypatch.delenv(OPENROUTER_API_KEY_ENV, raising=False)
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
        'name = "router"\n'
        'kind = "openrouter"\n'
        'model = "z-ai/glm-4.6"\n'
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
    assert "'router'" in flat and "kind=openrouter" in flat
    assert "USD(est)" not in flat  # refused before the cost confirmation
    assert seams.world_model.tasks == []  # zero cells ran, so `cheap` was never paid for
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


def test_route_sweep_persists_the_world_models_own_spend_as_a_run_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The line "metered separately" now has somewhere to point.

    The world model opens one metered session per episode and `WorldModelEnv.close` leaves that
    session's final record on the env; before this every one of them died there, so the sweep
    said the simulator's cost was accounted for elsewhere and nothing anywhere held it. They roll
    into one `kind="sweep"` record in the model's own runs dir, beside build and serve.
    """
    _patch_seams(monkeypatch, session_usd=0.03)
    root = _project(tmp_path, traces=_corpus())
    _out, result = _sweep(tmp_path, root, "support", "--scenarios", "2", "--yes")
    assert result.exit_code == 0, result.output

    sweeps = [r for r in load_runs(root / "models" / "support" / "runs") if r.kind == "sweep"]
    assert len(sweeps) == 1
    # 2 candidates x 2 scenarios x 1 episode = 4 sessions at $0.03 each.
    assert sweeps[0].total.cost_usd == pytest.approx(0.12)
    assert sweeps[0].by_phase[Phase.SERVE].calls == 4

    # Both sides are printed, and the candidate line is untouched: they are different money and
    # a single blended number would misprice both.
    assert _says(result.output, "measured candidate spend")
    assert _says(result.output, "measured world-model spend $0.1200 over 4 session(s)")
    assert _says(result.output, "eval infrastructure, not serving cost")


def test_route_fit_writes_compression_config(tmp_path: Path) -> None:
    config = CompressionConfig(compressor_id="truncate", aggressiveness=0.5)
    matrix_file = _matrix_file(tmp_path, compression=config)
    policy_file = tmp_path / "policy.json"
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(matrix_file),
            "--kind",
            "rank",
            "--out",
            str(policy_file),
            "--clusters",
            "2",
            "--dim",
            "64",
            "--compressor",
            "truncate",
            "--aggressiveness",
            "0.5",
        ],
    )
    assert result.exit_code == 0, result.output
    policy = RoutingPolicy.load(policy_file)
    assert policy.compression is not None
    assert policy.compression.compressor_id == "truncate"
    assert policy.compression.aggressiveness == 0.5


def test_route_fit_rejects_unknown_compressor(tmp_path: Path) -> None:
    matrix_file = _matrix_file(tmp_path)
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(matrix_file),
            "--out",
            str(tmp_path / "policy.json"),
            "--compressor",
            "llmzip",
        ],
    )
    assert result.exit_code != 0
    assert "unknown compressor" in result.output


def test_route_fit_rejects_orphan_aggressiveness(tmp_path: Path) -> None:
    matrix_file = _matrix_file(tmp_path)
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(matrix_file),
            "--out",
            str(tmp_path / "policy.json"),
            "--aggressiveness",
            "0.5",
        ],
    )
    assert result.exit_code != 0
    assert "--compressor" in result.output


def test_route_fit_stamps_what_the_evidence_was_fitted_under(tmp_path: Path) -> None:
    # D-COMPRESS requirement A: --compressor does not just switch serving on, it moves the FIT
    # onto the compressed representation and records that on the artifact, which is what makes
    # the resulting policy mountable at all.
    policy_file = tmp_path / "policy.json"
    config = CompressionConfig(compressor_id="truncate", aggressiveness=0.5)
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(_knn_matrix_file(tmp_path, compression=config)),
            "--kind",
            "knn",
            "--out",
            str(policy_file),
            "--dim",
            "64",
            "--compressor",
            "truncate",
            "--aggressiveness",
            "0.5",
        ],
    )
    assert result.exit_code == 0, result.output
    policy = RoutingPolicy.load(policy_file)  # loads, so the mount gate is satisfied
    assert policy.fit_compression is not None
    assert policy.fit_compression == policy.compression


def test_route_fit_refuses_to_stamp_an_arm_the_matrix_never_measured(tmp_path: Path) -> None:
    # The contract gap: --compressor moved the fit-side embeddings but could not retroactively
    # change what the EPISODES ran under, so a compressed policy could be stamped over rewards
    # measured uncompressed. That is a joint fit over an arm nobody ran.
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(_matrix_file(tmp_path)),  # uncompressed rewards
            "--out",
            str(tmp_path / "policy.json"),
            "--compressor",
            "truncate",
            "--aggressiveness",
            "0.5",
        ],
    )
    assert result.exit_code != 0
    assert "measured with raw text" in result.output
    assert "one matrix per arm" in result.output


def test_route_fit_refuses_compressed_rewards_under_a_raw_fit(tmp_path: Path) -> None:
    # The mirror image, and the same error: rewards produced under compression do not describe
    # an endpoint that serves raw text.
    config = CompressionConfig(compressor_id="truncate", aggressiveness=0.5)
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(_matrix_file(tmp_path, compression=config)),
            "--out",
            str(tmp_path / "policy.json"),
        ],
    )
    assert result.exit_code != 0
    flat = "".join(ch for ch in result.output if not ch.isspace() and ch not in "│┌┐└┘─╔╗╚╝║═")
    assert "wouldstamprawtext" in flat


def test_route_sweep_rejects_an_unservable_compressor_before_spending(tmp_path: Path) -> None:
    # The arm has to be one that could actually be served, and the check lands before any
    # episode is paid for.
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "sweep",
            "support",
            "--compressor",
            "llmzip",
            "--root",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
    assert "unknown compressor" in result.output


def test_route_sweep_rejects_orphan_aggressiveness(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "sweep",
            "support",
            "--aggressiveness",
            "0.5",
            "--root",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
    assert "--compressor" in result.output


def test_route_fit_stamps_the_running_compressor_version(tmp_path: Path) -> None:
    # Regression: the config was built without a version, so it defaulted to "1" and any fit
    # against a version-bumped compressor stamped a lie. The mount gate then hard-stopped the
    # result with a remedy the CLI has no flag to carry out. Latent while everything is v1,
    # which is exactly why it needs a test.
    class _V3(TruncateCompressor):
        id = "cli-v3-for-tests"
        version = "3"

    register_compressor(cast("Compressor", _V3()))
    config = CompressionConfig(
        compressor_id="cli-v3-for-tests", compressor_version="3", aggressiveness=0.5
    )
    policy_file = tmp_path / "policy.json"
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(_matrix_file(tmp_path, compression=config)),
            "--out",
            str(policy_file),
            "--dim",
            "64",
            "--compressor",
            "cli-v3-for-tests",
            "--aggressiveness",
            "0.5",
        ],
    )
    assert result.exit_code == 0, result.output
    policy = RoutingPolicy.load(policy_file)  # loads, so the version gate is satisfied
    assert policy.compression is not None
    assert policy.compression.compressor_version == "3"
    assert policy.fit_compression == policy.compression
    assert policy.serving_compressor() is not None  # and it mounts


def test_route_fit_knn_stamps_the_compression_it_was_fitted_under(tmp_path: Path) -> None:
    """The knn path must carry the stamp, not just the rank path.

    `--kind knn` returns from `fit_knn_artifact` before the rank path's stamping line, so a fit
    that attached compression only after the branch would write a knn policy with no
    `fit_compression` at all. That is the representation-consistency failure in a new costume:
    the endpoint would serve compressed while its bank claimed to be raw, and the mount gate
    would have nothing to compare.
    """
    config = CompressionConfig(compressor_id="truncate", aggressiveness=0.5)
    matrix_file = _knn_matrix_file(tmp_path, compression=config)
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
            "--out",
            str(policy_file),
            "--fallback",
            "a",
            "--min-pairs",
            "0",
            "--compressor",
            "truncate",
            "--aggressiveness",
            "0.5",
        ],
    )
    assert result.exit_code == 0, result.output
    policy = RoutingPolicy.load(policy_file)
    assert policy.kind == "knn"
    assert policy.compression is not None
    assert policy.compression.compressor_id == "truncate"
    # Both halves, so the mount gate has an identity to check rather than a null.
    assert policy.fit_compression is not None
    assert same_compression(policy.compression, policy.fit_compression)


def test_route_fit_knn_leaves_the_stamp_null_without_the_flag(tmp_path: Path) -> None:
    """The negative control: an uncompressed knn fit is byte-identical to before this seam."""
    policy_file = tmp_path / "policy.json"
    result = _fit_knn(_knn_matrix_file(tmp_path), policy_file)
    assert result.exit_code == 0, result.output
    policy = RoutingPolicy.load(policy_file)
    assert policy.compression is None and policy.fit_compression is None


# ------------------------------------------- input validation at the fit/report boundary
# `fit` and `report` used to hand user-typed paths straight to the pydantic and pathlib loaders,
# so a missing file, a swapped pair of positionals, or a matrix nothing scored came out as a
# Python traceback. Every test below asserts the same two things the rest of this file asserts
# for every other input: a clean usage error, and no exception escaping.


def _no_traceback(result: Result) -> bool:
    return result.exception is None or isinstance(result.exception, SystemExit)


def _unscored_matrix_file(tmp_path: Path) -> Path:
    """What a sweep writes when every episode errored: rows on disk, not one reward.

    `sweep` still saves this matrix (the cells were paid for and their `error` fields are the
    diagnosis) and exits 1 saying "fitting will fail", so it is a state a user reaches `fit`
    from rather than an invented one.
    """
    matrix = OutcomeMatrix.load(_matrix_file(tmp_path))
    path = tmp_path / "unscored.json"
    OutcomeMatrix(
        pool=matrix.pool,
        outcomes=[o.model_copy(update={"reward": None, "success": False}) for o in matrix.outcomes],
    ).save(path)
    return path


def test_route_fit_names_the_producer_when_the_matrix_is_missing(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(tmp_path / "nope.json"),
            "--embedder",
            "hashing",
            "--out",
            str(tmp_path / "policy.json"),
        ],
    )
    assert result.exit_code != 0
    assert _no_traceback(result)
    assert _says(result.output, "no outcome matrix at")
    assert _says(result.output, "wmo optimize route sweep")


def test_route_fit_rejects_a_matrix_that_is_not_readable(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(bad),
            "--embedder",
            "hashing",
            "--out",
            str(tmp_path / "policy.json"),
        ],
    )
    assert result.exit_code != 0
    assert _no_traceback(result)
    assert _says(result.output, "is not a readable OutcomeMatrix")


@pytest.mark.parametrize("kind", ["knn", "rank"])
def test_route_fit_refuses_a_matrix_with_no_scored_cell(tmp_path: Path, kind: str) -> None:
    """Both kinds, and both used to traceback.

    The rank fitter's own message ("no scored outcomes; cannot pick a default model") named no
    remedy at all, so the answer belongs at the boundary where sweep's warning can be echoed.
    """
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(_unscored_matrix_file(tmp_path)),
            "--kind",
            kind,
            "--embedder",
            "hashing",
            "--out",
            str(tmp_path / "policy.json"),
        ],
    )
    assert result.exit_code != 0
    assert _no_traceback(result)
    assert _says(result.output, "carries a reward")
    assert _says(result.output, "wmo optimize route sweep")
    assert not (tmp_path / "policy.json").exists()


def test_route_fit_defaults_to_the_knn_champion(tmp_path: Path) -> None:
    """The default kind has to be the one every other surface steers to.

    `fit --help` calls knn "the validated champion", sweep's handoff prints `--kind knn`, and
    `tune` only dials a knn policy -- but the flag defaulted to rank, so a user who omitted it
    silently fitted the non-champion and only found out at `tune`.
    """
    policy_file = tmp_path / POLICY_FILENAME
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(_knn_matrix_file(tmp_path)),
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
    assert RoutingPolicy.load(policy_file).kind == "knn"
    # And therefore dialable without a refit, which is what the old default was not.
    tuned = runner.invoke(
        app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "0.5"]
    )
    assert tuned.exit_code == 0, tuned.output


@pytest.mark.parametrize("command", ["fit", "sweep"])
def test_route_compressor_help_lists_every_shipped_id(command: str) -> None:
    """Rendered from the registry: `llmlingua2-endpoint` shipped while the help said two ids.

    Asserted against the ids registered at import rather than the live registry, because that is
    when typer builds a help string (this module itself registers fakes afterwards).
    """
    result = runner.invoke(app, ["optimize", "route", command, "--help"])
    assert result.exit_code == 0, result.output
    for compressor_id in ("identity", "truncate", "llmlingua2-endpoint"):
        assert compressor_id in registered_compressor_ids()
        assert _says(result.output, compressor_id)


def _report(matrix_file: Path, policy_file: Path, out: Path, *, baseline: str = "a") -> Result:
    """`route report`, always with an explicit --out so nothing lands in the working dir."""
    return runner.invoke(
        app,
        [
            "optimize",
            "route",
            "report",
            str(matrix_file),
            str(policy_file),
            "--baseline",
            baseline,
            "--out",
            str(out),
        ],
    )


def test_route_report_names_the_swap_when_the_positionals_are_reversed(tmp_path: Path) -> None:
    # Two same-typed positionals in a fixed order is a swap waiting to happen, and a pydantic
    # schema dump ("outcomes / Field required") is not a diagnosis.
    matrix_file = _knn_matrix_file(tmp_path)
    policy_file = _fitted_knn_policy(tmp_path)

    swapped = _report(policy_file, matrix_file, tmp_path / "report.json")
    assert swapped.exit_code != 0
    assert _no_traceback(swapped)
    assert _says(swapped.output, "holds a fitted policy, not an outcome matrix")
    assert _says(swapped.output, "wmo optimize route report <matrix.json> <policy.json>")
    assert not (tmp_path / "report.json").exists()


def test_route_report_rejects_a_policy_that_is_not_readable(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    result = _report(_knn_matrix_file(tmp_path), bad, tmp_path / "report.json")
    assert result.exit_code != 0
    assert _no_traceback(result)
    assert _says(result.output, "is not a readable routing policy")


def test_route_report_rejects_a_policy_that_is_not_utf8_text(tmp_path: Path) -> None:
    """`RoutingPolicy.load` decodes before pydantic runs, so this never reached the other clause.

    UnicodeDecodeError is a ValueError but NOT a ValidationError, so undecodable bytes (a
    truncated download, or the `.npz` evidence bank handed over as the policy) walked straight
    past the boundary and tracebacked.
    """
    bad = tmp_path / "bad.json"
    bad.write_bytes(b'{"kind": "\xff\xfeknn"}')
    result = _report(_knn_matrix_file(tmp_path), bad, tmp_path / "report.json")
    assert result.exit_code != 0
    assert _no_traceback(result)
    assert _says(result.output, "cannot read the policy at")
    assert not (tmp_path / "report.json").exists()


def test_route_report_delivers_the_missing_sidecar_message_cleanly(tmp_path: Path) -> None:
    """The message was already written; it arrived as the last line of a stack trace.

    Copying a knn policy.json without its `.bank.npz` is the exact mistake `knn_bank` anticipates.
    """
    matrix_file = _knn_matrix_file(tmp_path)
    policy_file = _fitted_knn_policy(tmp_path)
    RoutingPolicy.load(policy_file).bank_path().unlink()

    result = _report(matrix_file, policy_file, tmp_path / "report.json")
    assert result.exit_code != 0
    assert _no_traceback(result)
    assert _says(result.output, "knn policy bank not found at")
    assert _says(result.output, "Copy the sidecar next to the policy file")


def test_route_report_says_baseline_is_a_pool_handle(tmp_path: Path) -> None:
    # `--baseline` takes the [[model]] table's `name`, not the model id, and passing an id used
    # to raise a bare KeyError.
    result = _report(
        _knn_matrix_file(tmp_path),
        _fitted_knn_policy(tmp_path),
        tmp_path / "report.json",
        baseline="gpt-4o",
    )
    assert result.exit_code != 0
    assert _no_traceback(result)
    assert _says(result.output, "baseline 'gpt-4o' is not in the matrix pool")
    assert _says(result.output, "pool entry handle")


def test_route_report_refuses_a_matrix_with_nothing_scored_on_both_sides(tmp_path: Path) -> None:
    matrix = OutcomeMatrix.load(_matrix_file(tmp_path))
    half = tmp_path / "half.json"
    OutcomeMatrix(
        pool=matrix.pool,
        outcomes=[
            o.model_copy(update={"reward": None, "success": False}) if o.model == "a" else o
            for o in matrix.outcomes
        ],
    ).save(half)
    policy_file = tmp_path / "static.json"
    RoutingPolicy(kind="static", default_model="a", pool=matrix.pool, fitted_from="handmade").save(
        policy_file
    )

    result = _report(half, policy_file, tmp_path / "report.json", baseline="b")
    assert result.exit_code != 0
    assert _no_traceback(result)
    assert _says(result.output, "nothing to compare")
    assert not (tmp_path / "report.json").exists()


def test_route_report_creates_the_out_directory_like_fit_does(tmp_path: Path) -> None:
    """`fit --out` mkdir -p's its parents; report tracebacked AFTER computing the whole report."""
    out = tmp_path / "missing" / "sub" / "report.json"
    result = _report(_knn_matrix_file(tmp_path), _fitted_knn_policy(tmp_path), out)
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text(encoding="utf-8"))["headline"]


def test_route_report_notes_the_excluded_fit_split_on_the_fit_matrix(
    tmp_path: Path,
) -> None:
    """Same matrix as the fit: since #308 the report excludes the fit split, so the surface says
    "held-out with N fit scenarios excluded" rather than contradicting the report's own label.
    The matrix digest in `fitted_from` is an identity, so a renamed copy has to trip it too.
    """
    matrix_file = _knn_matrix_file(tmp_path)
    policy_file = _fitted_knn_policy(tmp_path)

    result = _report(matrix_file, policy_file, tmp_path / "report.json")
    assert result.exit_code == 0, result.output
    assert _says(result.output, "fit scenario(s) were excluded")
    assert "IN-SAMPLE" not in _flat(result.output)

    renamed = tmp_path / "renamed.json"
    renamed.write_bytes(matrix_file.read_bytes())
    moved = _report(renamed, policy_file, tmp_path / "report_renamed.json")
    assert moved.exit_code == 0, moved.output
    assert _says(moved.output, "fit scenario(s) were excluded")

    # The provenance marker is appended LAST, so a matrix stored under a content-addressed
    # directory carries `sha256=` in its path too. Splitting from the left read THAT one and
    # dropped the caveat on exactly the layout most likely to keep a fit matrix around.
    addressed = tmp_path / "artifacts" / "sha256=deadbeef" / "matrix.json"
    addressed.parent.mkdir(parents=True)
    addressed.write_bytes(matrix_file.read_bytes())
    content_addressed = _report(addressed, policy_file, tmp_path / "report_addressed.json")
    assert content_addressed.exit_code == 0, content_addressed.output
    assert _says(content_addressed.output, "fit scenario(s) were excluded")


def test_route_report_stays_quiet_on_a_matrix_the_fit_never_saw(tmp_path: Path) -> None:
    """The negative control: held-out numbers are what report is for, so no warning."""
    policy_file = _fitted_knn_policy(tmp_path)
    held_out = _knn_matrix_file(tmp_path, flip=True, name="held_out.json")
    result = _report(held_out, policy_file, tmp_path / "report.json")
    assert result.exit_code == 0, result.output
    assert "IN-SAMPLE" not in _flat(result.output)


def test_route_pin_names_the_positional_when_the_model_is_ambiguous(tmp_path: Path) -> None:
    # `WorldModelStore.resolve` says "pass --name", which `pin` does not have: its world model is
    # a positional and its --model is the POOL entry. Following the old advice failed outright.
    pool_file = tmp_path / "pool.toml"
    assert _add_student(tmp_path, pool_file).exit_code == 0
    _built_model(tmp_path, "alpha")
    _built_model(tmp_path, "beta")

    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "pin",
            "--model",
            "student",
            "--pool",
            str(pool_file),
            "--root",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
    flat = _flat(result.output)
    assert "WORLD_MODEL" in flat and "alpha,beta" in flat
    assert "--name" not in flat
    assert "wmooptimizeroutepinalpha--modelstudent" in flat  # a command that actually works

    followed = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "pin",
            "alpha",
            "--model",
            "student",
            "--pool",
            str(pool_file),
            "--root",
            str(tmp_path),
        ],
    )
    assert followed.exit_code == 0, followed.output


def test_route_pin_names_the_pool_writers_when_the_pool_is_empty(tmp_path: Path) -> None:
    pool_file = tmp_path / "pool.toml"
    pool_file.write_text("# nothing yet\n", encoding="utf-8")
    _built_model(tmp_path)

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
    assert result.exit_code != 0
    assert _no_traceback(result)
    assert _says(result.output, str(pool_file))
    assert _says(result.output, "wmo optimize route student")
    assert "too_short" not in _flat(result.output)  # was a raw pydantic dump


def test_route_tune_names_the_fit_when_there_is_no_policy(tmp_path: Path) -> None:
    # The sibling not-dialable error names `wmo optimize route fit --kind knn`; this branch,
    # which is the one a first-time user hits (the argument defaults to ./policy.json), did not.
    result = runner.invoke(
        app, ["optimize", "route", "tune", str(tmp_path / "nope.json"), "--cost-quality", "0.5"]
    )
    assert result.exit_code != 0
    assert _says(result.output, "no policy file at")
    assert _says(result.output, "wmo optimize route fit <matrix.json> --kind knn")


# ------------------------------------------- rich markup in help text


def test_route_student_help_keeps_the_pool_table_name() -> None:
    """The paragraph exists to name the TOML table `student` writes, so it must survive rich.

    Typer renders help through rich markup, which swallowed the unescaped `[[model]]` and left
    an empty pair of backticks where the identifier should be.
    """
    result = runner.invoke(app, ["optimize", "route", "student", "--help"])
    assert result.exit_code == 0, result.output
    assert "[[model]]" in _flat(result.output)


def test_route_pin_refuses_a_disabled_model_and_drops_disabled_entries_from_the_pool(
    tmp_path: Path,
) -> None:
    """`enabled = false` is honored at pin time: not pinnable, and not carried into the policy.

    The policy's pool is what serving may construct providers for, so a candidate the operator
    turned off must not ride into an endpoint pinned afterwards.
    """
    pool_file = tmp_path / "pool.toml"
    assert _add_student(tmp_path, pool_file).exit_code == 0
    pool_file.write_text(
        pool_file.read_text(encoding="utf-8")
        + """
[[model]]
name = "off-limits"
kind = "openai"
model = "gpt-5.4"
enabled = false
""",
        encoding="utf-8",
    )
    model_dir = _built_model(tmp_path)

    refused = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "pin",
            "support",
            "--model",
            "off-limits",
            "--pool",
            str(pool_file),
            "--root",
            str(tmp_path),
        ],
    )
    assert refused.exit_code != 0
    assert "disabled" in refused.output

    pinned = runner.invoke(
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
    assert pinned.exit_code == 0, pinned.output
    policy = RoutingPolicy.load(model_dir / POLICY_FILENAME)
    assert [entry.name for entry in policy.pool] == ["student"]


def test_route_student_replacement_keeps_a_disabled_entry_disabled(tmp_path: Path) -> None:
    """Retraining and re-registering a student must not undo an operator's enabled = false.

    Same rule as the registry writer, pinned separately because the student command
    reaches upsert_pool_entry through its own path (_pool_disabled at route_app.py).
    """
    pool_file = tmp_path / "pool.toml"
    assert _add_student(tmp_path, pool_file).exit_code == 0
    pool_file.write_text(
        pool_file.read_text(encoding="utf-8").replace(
            'name = "student"', 'name = "student"\nenabled = false'
        ),
        encoding="utf-8",
    )

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
    assert "keeping it disabled" in result.output
    entries = load_pool(pool_file).models
    assert len(entries) == 1
    assert entries[0].enabled is False


def _fitted_knn_policy(tmp_path: Path) -> Path:
    """Fit a real knn policy + sidecar through the CLI, returning the policy path."""
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
    return policy_file


class _FakeClient:
    """Records the install call instead of making it."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Path, Path | None, Path | None]] = []

    def install_endpoint_policy(
        self,
        org_id: str,
        endpoint: str,
        policy_path: Path,
        bank_path: Path | None,
        report_path: Path | None = None,
    ) -> dict[str, str]:
        self.calls.append((org_id, endpoint, policy_path, bank_path, report_path))
        return {"name": endpoint}


@contextmanager
def _connected_to(client: _FakeClient) -> Iterator[None]:
    """Stand in for a platform login for the duration of one command.

    Patches `platform_cmds`, which is where `push` resolves the connection from, so
    the command runs its real body against a client that records instead of calling.
    """
    import wmo.cli.platform_cmds as platform_module

    @contextmanager
    def _fake_connected(_credentials: object, _headline: str) -> Iterator[_FakeClient]:
        yield client

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(platform_module, "_connected", _fake_connected)
        patch.setattr(platform_module, "_require_connection", lambda _org: (None, "org-1"))
        yield


def test_route_push_sends_the_policy_and_its_sidecar(tmp_path: Path) -> None:
    """Push sends BOTH artifacts: a knn policy alone would store an unservable row."""
    policy_file = _fitted_knn_policy(tmp_path)
    client = _FakeClient()
    with _connected_to(client):
        result = runner.invoke(
            app,
            ["optimize", "route", "push", str(policy_file), "--endpoint", "support-prod"],
        )
    assert result.exit_code == 0, result.output
    assert len(client.calls) == 1
    _org, endpoint, sent_policy, sent_bank, sent_report = client.calls[0]
    assert endpoint == "support-prod"
    assert sent_policy == policy_file
    # Resolved from the policy's own knn_bank_path, not guessed from the policy name.
    assert sent_bank == RoutingPolicy.load(policy_file).bank_path()
    assert sent_bank is not None and sent_bank.is_file()
    assert sent_report is None
    assert _says(result.output, "installed knn policy")


def test_route_push_refuses_a_knn_policy_whose_sidecar_is_missing(tmp_path: Path) -> None:
    """A knn policy without its bank fails locally, before any upload."""
    # The pair is broken on this machine, so pushing would only turn a local mistake
    # into a server refusal after sending a policy the server must reject.
    policy_file = _fitted_knn_policy(tmp_path)
    RoutingPolicy.load(policy_file).bank_path().unlink()
    client = _FakeClient()
    with _connected_to(client):
        result = runner.invoke(
            app,
            ["optimize", "route", "push", str(policy_file), "--endpoint", "support-prod"],
        )
    assert result.exit_code != 0
    assert _says(result.output, "evidence bank is missing")
    assert client.calls == []


def test_route_push_refuses_a_path_that_is_not_a_policy(tmp_path: Path) -> None:
    """A file that is not a routing policy is refused without contacting the platform."""
    junk = tmp_path / "policy.json"
    junk.write_text('{"kind": "not-a-kind"}', encoding="utf-8")
    client = _FakeClient()
    with _connected_to(client):
        result = runner.invoke(
            app, ["optimize", "route", "push", str(junk), "--endpoint", "support-prod"]
        )
    assert result.exit_code != 0
    assert _says(result.output, "not a routing policy")
    assert client.calls == []


def test_route_push_sends_no_bank_for_a_static_policy(tmp_path: Path) -> None:
    """A static policy has no sidecar, so none is sent and none is required."""
    policy_file = tmp_path / "policy.json"
    RoutingPolicy(
        kind="static",
        default_model="a",
        pool=[
            PoolEntry(
                name="a",
                kind=ProviderKind.OPENAI,
                model="a",
                input_per_mtok=1.0,
                output_per_mtok=1.0,
            )
        ],
    ).save(policy_file)
    client = _FakeClient()
    with _connected_to(client):
        result = runner.invoke(
            app, ["optimize", "route", "push", str(policy_file), "--endpoint", "support-prod"]
        )
    assert result.exit_code == 0, result.output
    assert client.calls[0][3] is None


def test_route_push_sends_the_report_when_given_one(tmp_path: Path) -> None:
    """--report rides through to the install, so the endpoint gains its evidence."""
    policy_file = _fitted_knn_policy(tmp_path)
    report_file = tmp_path / "report.json"
    report_file.write_text('{"headline": {}}', encoding="utf-8")
    client = _FakeClient()
    with _connected_to(client):
        result = runner.invoke(
            app,
            [
                "optimize",
                "route",
                "push",
                str(policy_file),
                "--endpoint",
                "support-prod",
                "--report",
                str(report_file),
            ],
        )
    assert result.exit_code == 0, result.output
    assert client.calls[0][4] == report_file
