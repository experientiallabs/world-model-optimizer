"""CLI tests for `wmo optimize model`, driven via CliRunner with every provider stubbed.

Zero real LLM calls and zero spend: the world model is a canned-score stub and every pool
candidate is a two-line script, so a full preflight -> sweep -> fit -> tune -> report run
happens in milliseconds against the real code path.
"""

from __future__ import annotations

import importlib
import json
import re
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
from wmo.engine.world_model import WorldModel
from wmo.ingest.otel_writer import write_traces_jsonl
from wmo.optimize.outcomes import OutcomeMatrix
from wmo.optimize.pipeline import (
    MANIFEST_FILENAME,
    MATRIX_FILENAME,
    REPORT_FILENAME,
    RunManifest,
    Stage,
)
from wmo.optimize.policy import POLICY_FILENAME, RoutingPolicy
from wmo.optimize.report import ImprovementReport
from wmo.optimize.reward import EpisodeScore
from wmo.optimize.sweep import SweepPlan
from wmo.providers.base import (
    Completion,
    Message,
    Provider,
    ProviderConfig,
    ProviderKind,
    TokenUsage,
    VerifyResult,
)
from wmo.serving.traces_source import TRACES_FILENAME
from wmo.tracking import Phase, RunRecord, UsageTotals, load_runs

runner = CliRunner()

optimize_module = importlib.import_module("wmo.cli.optimize_model_app")
route_module = importlib.import_module("wmo.cli.route_app")

_HELD_OUT_IDS = ("tr-010", "tr-018", "tr-020", "tr-027")
_FRAME_CHARS = frozenset("│┃╭╮╰╯─━┏┓┗┛┡┩┢┪╇╈├┤┬┴┼")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _flat(text: str) -> str:
    """Text with color, whitespace, and rich's box-drawing frame removed.

    Rich wraps (and frames, and at a forced terminal colorizes) everything it prints, so a
    literal substring check against the raw output is a coin flip on where the wrap landed and
    whether the highlighter split a number out of its sentence.
    """
    plain = _ANSI.sub("", text)
    return "".join(ch for ch in plain if not ch.isspace() and ch not in _FRAME_CHARS)


def _says(output: str, phrase: str) -> bool:
    return _flat(phrase) in _flat(output)


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """Render the plan table wide enough that no cell wraps.

    Rich lays a table out line by line, so a wrapped cell interleaves its continuation with the
    NEXT column's text: at 80 columns "SKIP (matrix.json is current: ...)" comes back cut in
    three and spliced through the plan column, and every assertion on a printed sentence becomes
    an assertion about where the wrap landed. Width is presentation, not behavior.
    """
    monkeypatch.setattr(optimize_module, "_console", Console(width=240))


def _corpus(count: int = 30) -> list[Trace]:
    """A corpus whose deterministic split leaves a real held-out band, one task per trace."""
    return [
        Trace(
            trace_id=f"tr-{index:03d}",
            steps=[
                Step(
                    action=Action(kind=ActionKind.TOOL_CALL, name="ls", arguments={"path": "."}),
                    observation=Observation(content="a.txt"),
                    task=f"task tr-{index:03d}",
                ),
                Step(
                    action=Action(kind=ActionKind.MESSAGE, content="done"),
                    observation=Observation(content="ok"),
                    task=f"task tr-{index:03d}",
                ),
            ],
        )
        for index in reversed(range(count))
    ]


def _project(tmp_path: Path) -> Path:
    """A built-model artifact dir (config + its own corpus); returns the project root."""
    root = tmp_path / ".wmo"
    model_dir = root / "models" / "support"
    save_config(
        HarnessConfig(
            providers=[ProviderConfig(kind=ProviderKind.ANTHROPIC, model="fake-serve")],
            serve_provider=ProviderKind.ANTHROPIC,
            train_split=0.8,
        ),
        model_dir,
    )
    write_traces_jsonl(_corpus(), model_dir / TRACES_FILENAME)
    return root


def _pool_file(tmp_path: Path, *, pricey_out: float = 20.0) -> Path:
    """Two priced candidates, 1/2 and 10/`pricey_out` USD per Mtok, so costs are distinguishable."""
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
        f"output_per_mtok = {pricey_out}\n",
        encoding="utf-8",
    )
    return path


class _FakeWorldModel:
    """`WorldModel`-shaped stub: in-memory sessions, a canned episode score, no LLM at all."""

    def __init__(self, rewards: dict[str, float] | None = None, session_usd: float = 0.02) -> None:
        # Per-candidate rewards keyed by the model id the episode's provider was built with, so a
        # sweep can produce a matrix where one candidate is genuinely better than another.
        self._rewards = rewards or {}
        # What the simulator charges per episode for its OWN serve + judge calls: the
        # world-model side of a sweep's bill, which is metered separately from the candidates.
        self._session_usd = session_usd
        self._frozen = False
        self.tasks: list[str | None] = []
        self.current_model = "cheap-1"

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
        return Session(id=f"s{len(self.tasks)}", task=task, enrich=enrich)

    def step(self, session_id: str, action: Action) -> Observation:
        return Observation(content="ok")

    def score_session(self, session_id: str) -> EpisodeScore:
        reward = self._rewards.get(self.current_model, 0.75)
        return EpisodeScore(reward=reward, success=reward >= 0.5, critique="fine")

    def end_session(self, session_id: str) -> RunRecord:
        return self._usage_record(session_id)

    def session_usage(self, session_id: str) -> RunRecord:
        return self._usage_record(session_id)

    def _usage_record(self, session_id: str) -> RunRecord:
        serve = UsageTotals(
            calls=2, input_tokens=400, output_tokens=60, cost_usd=self._session_usd * 0.75
        )
        judge = UsageTotals(
            calls=1, input_tokens=200, output_tokens=30, cost_usd=self._session_usd * 0.25
        )
        return RunRecord(
            run_id=session_id,
            kind="serve",
            duration_seconds=0.5,
            total=serve.merged(judge),
            by_phase={Phase.SERVE: serve, Phase.JUDGE: judge},
        )


class _ScriptedCandidate:
    """A candidate that calls one tool and then declares itself done.

    `throttled` makes every completion raise instead, the way a rate-limited candidate does: the
    episode errors, `run_episode` records it, and the cell comes back unscored.
    """

    def __init__(
        self, config: ProviderConfig, world_model: _FakeWorldModel, *, throttled: bool = False
    ) -> None:
        self.config = config
        self._world_model = world_model
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
        if self._throttled:
            raise RuntimeError("rate limit exceeded (429)")
        # The env scores on close, after this provider's last call, so recording which candidate
        # is live here is what lets the fake judge give different candidates different rewards.
        self._world_model.current_model = self.config.model
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
        self.asked: list[str] = []

    def ask(self, prompt: str, *, default: bool = True) -> bool:
        self.asked.append(prompt)
        return self._answer


def _patch_seams(
    monkeypatch: pytest.MonkeyPatch,
    *,
    rewards: dict[str, float] | None = None,
    modules: tuple[object, ...] = (),
    session_usd: float = 0.02,
    throttled_models: frozenset[str] = frozenset(),
) -> _FakeWorldModel:
    """Stub the world model and every pool provider; return the fake for post-run assertions."""
    world_model = _FakeWorldModel(rewards=rewards, session_usd=session_usd)

    def _load(model_dir: Path) -> tuple[WorldModel, Provider]:
        provider = _ScriptedCandidate(
            ProviderConfig(kind=ProviderKind.ANTHROPIC, model="fake-serve"), world_model
        )
        return cast("WorldModel", world_model), cast("Provider", provider)

    def _get_provider(config: ProviderConfig, api_key: str | None = None) -> Provider:
        return cast(
            "Provider",
            _ScriptedCandidate(config, world_model, throttled=config.model in throttled_models),
        )

    for module in (optimize_module, *modules):
        monkeypatch.setattr(module, "load_world_model", _load)
    monkeypatch.setattr("wmo.providers.pool.get_provider", _get_provider)
    return world_model


def _run(tmp_path: Path, root: Path, *extra: str, pool: Path | None = None) -> Result:
    """Invoke `wmo optimize model support` against the temp project."""
    return runner.invoke(
        app,
        [
            "optimize",
            "model",
            "support",
            "--root",
            str(root),
            "--pool",
            str(pool or _pool_file(tmp_path)),
            "--scenarios",
            "3",
            "--max-steps",
            "4",
            *extra,
        ],
    )


def _paths(root: Path) -> tuple[Path, Path, Path, Path]:
    """(matrix, policy, report, manifest) for the `support` model under `root`."""
    model_dir = root / "models" / "support"
    run_dir = model_dir / "optimize"
    return (
        run_dir / MATRIX_FILENAME,
        model_dir / POLICY_FILENAME,
        run_dir / REPORT_FILENAME,
        run_dir / MANIFEST_FILENAME,
    )


def test_one_command_lands_every_artifact_where_serving_reads_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The promise: after one command the model has a fitted, dialed, servable policy."""
    _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9})
    root = _project(tmp_path)
    result = _run(tmp_path, root, "--yes")
    assert result.exit_code == 0, result.output

    matrix_path, policy_path, report_path, manifest_path = _paths(root)
    # The matrix and the report are this command's own; the policy is where `wmo serve` looks.
    matrix = OutcomeMatrix.load(matrix_path)
    assert len(matrix.outcomes) == 6  # 2 candidates x 3 scenarios x 1 episode
    assert matrix.scenario_ids() == list(_HELD_OUT_IDS[:3])
    policy = RoutingPolicy.load(policy_path)
    assert policy.kind == "knn"
    assert policy.knn_bank_path is not None
    # tune's as-fitted snapshot semantics survive the orchestrator: the dial is recorded and the
    # un-tuned artifact is preserved beside it, so sliding again never compounds.
    assert policy.cost_quality == 0.25
    assert (policy_path.parent / "policy.base.json").is_file()
    assert RoutingPolicy.load(policy_path.parent / "policy.base.json").cost_quality is None
    report = ImprovementReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    assert report.endpoint_id == "support"
    assert report.headline.scenarios_compared == 3

    manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    assert [record.stage.value for record in manifest.stages] == ["sweep", "fit", "tune", "report"]
    assert manifest.world_model == "support"

    # The plan table printed every stage before anything spent, and the run ended on the payoff.
    assert _says(result.output, "optimize model: support")
    for stage in ("preflight", "sweep", "fit", "tune", "report"):
        assert stage in _flat(result.output)
    assert _says(result.output, "estimated candidate spend")
    assert _says(result.output, "serve it:   wmo serve --name support")
    assert _says(result.output, 'POST /v1/chat/completions  (model="support")')
    assert "quality" in _flat(result.output) and "latency" in _flat(result.output)


def test_a_second_run_skips_every_stage_and_says_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume is the property that makes the command safe to re-type: no cell is bought twice."""
    world_model = _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9})
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes").exit_code == 0
    episodes_after_first = len(world_model.tasks)

    again = _run(tmp_path, root, "--yes")
    assert again.exit_code == 0, again.output
    flat = _flat(again.output)
    assert len(world_model.tasks) == episodes_after_first  # not one new episode
    # Every skip states what was unchanged, not just that it skipped.
    assert _says(again.output, "matrix.json is current: same pool, same scenarios, same episodes")
    assert _says(again.output, "policy.json is current: same matrix, same knn knobs")
    assert _says(again.output, "dial already at 0.25")
    assert _says(again.output, "report.json is current")
    assert "everystageiscurrent" in flat
    # A run with nothing to do asks nothing and spends nothing.
    assert "estimatedcandidatespend" not in flat


def test_force_from_sweep_redoes_the_sweep_and_everything_after_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world_model = _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9})
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes").exit_code == 0
    before = len(world_model.tasks)

    forced = _run(tmp_path, root, "--yes", "--force-from", "sweep")
    assert forced.exit_code == 0, forced.output
    assert len(world_model.tasks) == before * 2  # the cells were bought again, as asked
    assert _says(forced.output, "forced by --force-from")
    # Downstream is redone because its input is about to change, and the table says exactly that.
    assert _says(forced.output, "runs after sweep, which will change its input")


def test_force_from_fit_leaves_the_paid_sweep_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The point of a staged redo: refitting must never re-buy cells.
    world_model = _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9})
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes").exit_code == 0
    before = len(world_model.tasks)

    forced = _run(tmp_path, root, "--yes", "--force-from", "fit")
    assert forced.exit_code == 0, forced.output
    assert len(world_model.tasks) == before
    assert _says(forced.output, "matrix.json is current")


def test_editing_the_pool_reruns_the_sweep_and_names_the_pool_as_the_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world_model = _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9})
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes").exit_code == 0
    before = len(world_model.tasks)

    repriced = _pool_file(tmp_path, pricey_out=30.0)
    changed = _run(tmp_path, root, "--yes", pool=repriced)
    assert changed.exit_code == 0, changed.output
    assert len(world_model.tasks) > before
    assert _says(changed.output, "pool changed")


def test_deleting_the_optimize_dir_resets_resume_without_touching_the_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No hidden state: the manifest dir is disposable and the serving artifact is not in it."""
    _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9})
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes").exit_code == 0
    matrix_path, policy_path, _report, manifest_path = _paths(root)
    fitted_before = RoutingPolicy.load(policy_path).fitted_from

    for path in (matrix_path, manifest_path, matrix_path.parent / REPORT_FILENAME):
        path.unlink()
    matrix_path.parent.rmdir()
    assert policy_path.is_file()  # serving still works, which is the whole point

    again = _run(tmp_path, root, "--yes")
    assert again.exit_code == 0, again.output
    assert _says(again.output, "never completed here")
    assert RoutingPolicy.load(policy_path).fitted_from != fitted_before  # a genuinely fresh fit


def test_a_corrupt_manifest_warns_and_replans_instead_of_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world_model = _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9})
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes").exit_code == 0
    before = len(world_model.tasks)
    _matrix, _policy, _report, manifest_path = _paths(root)
    manifest_path.write_text("{ truncated", encoding="utf-8")

    again = _run(tmp_path, root, "--yes")
    assert again.exit_code == 0, again.output
    assert _says(again.output, "could not be read as a run manifest")
    # Be honest about the price of a reset rather than claiming it is free: with no records to
    # match against, every stage reads as "never completed here" and the sweep is measured again.
    # `RunManifest.save` is atomic precisely so this path stays rare.
    assert _says(again.output, "never completed here")
    assert len(world_model.tasks) == before * 2  # re-bought, which is what a reset costs
    assert RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8")).stages


def test_distill_is_rejected_with_the_command_that_does_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_seams(monkeypatch)
    root = _project(tmp_path)
    result = _run(tmp_path, root, "--yes", "--distill", "distill.toml")
    assert result.exit_code != 0
    flat = _flat(result.output)
    assert "reservedandnotimplementedinthisbuild" in flat
    assert _says(result.output, "wmo optimize distill run")
    assert _says(result.output, "wmo optimize route student")
    assert not _paths(root)[0].exists()  # nothing ran


def test_force_from_a_reserved_stage_says_it_is_not_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_seams(monkeypatch)
    root = _project(tmp_path)
    result = _run(tmp_path, root, "--yes", "--force-from", "compact")
    assert result.exit_code != 0
    flat = _flat(result.output)
    assert "reservedslot" in flat and "sweep|fit|tune|report" in flat


def test_force_from_an_unknown_stage_lists_the_real_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_seams(monkeypatch)
    root = _project(tmp_path)
    result = _run(tmp_path, root, "--yes", "--force-from", "nonsense")
    assert result.exit_code != 0
    assert _says(result.output, "unknown stage 'nonsense'")
    assert "sweep|fit|tune|report" in _flat(result.output)


def test_the_spend_cap_stops_before_the_sweep_and_prints_how_to_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 2 candidates x 3 scenarios x 4 calls at the pool's prices projects well over $0.01.
    world_model = _patch_seams(monkeypatch)
    root = _project(tmp_path)
    result = _run(tmp_path, root, "--yes", "--max-usd", "0.01")
    assert result.exit_code == 1, result.output
    assert world_model.tasks == []  # stopped BEFORE the stage, not during it
    flat = _flat(result.output)
    assert "stoppedatthespendcap" in flat
    assert _says(result.output, "wmo optimize model support --max-usd <more>")
    assert not _paths(root)[0].exists()


def test_a_cap_that_covers_the_run_lets_it_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The negative control for the cap test: the same run with room under the cap completes.
    _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9})
    root = _project(tmp_path)
    result = _run(tmp_path, root, "--yes", "--max-usd", "100")
    assert result.exit_code == 0, result.output
    assert _paths(root)[0].is_file()


def test_declining_the_confirmation_spends_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world_model = _patch_seams(monkeypatch)
    root = _project(tmp_path)
    answer = _Answer(False)
    monkeypatch.setattr(optimize_module, "Confirm", answer)
    monkeypatch.setattr(optimize_module, "_console", Console(width=240, force_terminal=True))
    result = _run(tmp_path, root)
    assert result.exit_code == 0, result.output
    assert answer.asked  # exactly one question, and it was asked before any episode
    assert len(answer.asked) == 1
    assert world_model.tasks == []
    assert not _paths(root)[0].exists()


def test_the_plan_table_prices_the_sweep_and_labels_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_seams(monkeypatch)
    root = _project(tmp_path)
    answer = _Answer(False)
    monkeypatch.setattr(optimize_module, "Confirm", answer)
    sweep_plans: list[str] = []
    render_stage_plan = optimize_module._stage_plan_text

    def record_stage_plan(
        stage: Stage, *, plan: SweepPlan, cost_quality: float, fallback: str | None, anchor: str
    ) -> str:
        text = render_stage_plan(
            stage, plan=plan, cost_quality=cost_quality, fallback=fallback, anchor=anchor
        )
        if stage is Stage.SWEEP:
            sweep_plans.append(text)
        return text

    monkeypatch.setattr(optimize_module, "_stage_plan_text", record_stage_plan)
    result = _run(tmp_path, root)
    flat = _flat(result.output)
    # 3 scenarios x 1 episode x 4 calls = 12 calls; cheap = 12 x (2000 + 250x2)/1e6 = $0.03,
    # pricey = 10x that, so the projected total is $0.33.
    assert "~$0.33" in flat
    assert sweep_plans == ["2 candidate(s) x 3 scenario(s) x 1 episode(s)"]
    # The free stages say free rather than showing a fabricated number, and the estimate names
    # itself a projection with its assumption spelled out.
    assert _says(result.output, "knn (guarded, fallback best single on the sweep)")
    assert _says(result.output, "cost_quality 0.25 (Balanced (default))")
    assert "aprojection" in flat and "assumedoutputtoken" in flat
    assert _says(result.output, "are NOT in that figure")


def test_an_unscored_sweep_withholds_the_fit_and_keeps_the_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The orchestrator owns no coverage policy of its own: `route sweep`'s contract holds."""
    _patch_seams(monkeypatch)
    root = _project(tmp_path)
    real_env = optimize_module.WorldModelEnv
    monkeypatch.setattr(
        optimize_module,
        "WorldModelEnv",
        lambda world_model, *, score_on_close=False: real_env(world_model),
    )
    world_model = _patch_seams(monkeypatch)
    result = _run(tmp_path, root, "--yes")
    assert result.exit_code == 1, result.output
    matrix_path, policy_path, _report, manifest_path = _paths(root)
    assert matrix_path.is_file()  # the paid cells are on disk with their errors
    assert all(not outcome.scored for outcome in OutcomeMatrix.load(matrix_path).outcomes)
    assert not policy_path.exists()  # the fit was withheld
    assert _says(result.output, "no cell was scored")

    # ...and the rejected sweep is RECORDED, so the second attempt costs nothing. Before this was
    # fixed the contract's exit ran before the record was saved, and every retry re-bought every
    # cell while the printed message claimed otherwise.
    manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    assert manifest.record_for(Stage.SWEEP) is not None
    bought = len(world_model.tasks)
    again = _run(tmp_path, root, "--yes")
    assert again.exit_code == 1, again.output
    assert len(world_model.tasks) == bought  # not one cell re-bought
    assert _says(again.output, "no cell was scored")  # and still refused


def test_the_sweep_stage_and_route_sweep_produce_identical_matrices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The extraction's guarantee: two commands, one measurement, byte-identical evidence.

    Both faces call `wmo.optimize.sweep`, so a divergence here would mean one of them grew its
    own copy of the scenario cut, the tools hint, or the cell ordering.
    """
    _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9}, modules=(route_module,))
    root = _project(tmp_path)
    pool = _pool_file(tmp_path)

    assert _run(tmp_path, root, "--yes", pool=pool).exit_code == 0
    orchestrated = json.loads(_paths(root)[0].read_text(encoding="utf-8"))

    manual_out = tmp_path / "manual-matrix.json"
    manual = runner.invoke(
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
            str(manual_out),
            "--scenarios",
            "3",
            "--max-steps",
            "4",
            "--yes",
        ],
    )
    assert manual.exit_code == 0, manual.output
    manually = json.loads(manual_out.read_text(encoding="utf-8"))

    assert orchestrated["pool"] == manually["pool"]
    assert len(orchestrated["outcomes"]) == len(manually["outcomes"])
    # Compared as WHOLE cells minus a named exclusion, not as an allowlist of fields: an
    # allowlist silently stops covering anything added to ScenarioOutcome later, and cost and
    # error capture are exactly where two faces would drift.
    wall_clock = {"call_seconds"}
    for cell, other in zip(orchestrated["outcomes"], manually["outcomes"], strict=True):
        assert {k: v for k, v in cell.items() if k not in wall_clock} == {
            k: v for k, v in other.items() if k not in wall_clock
        }


def test_the_closing_numbers_name_their_anchor_and_their_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every headline number says what it was measured against and how (AGENTS numbers honesty)."""
    _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9})
    root = _project(tmp_path)
    result = _run(tmp_path, root, "--yes")
    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert "world-modelsimulated" in flat and "held-outscenario" in flat
    assert "measuredcandidate-sideatlistprices" in flat
    assert "cacheeffectsnotmodeled" in flat
    assert "walltimeperpolicycall" in flat
    # The dial and the fallback are named beside them, so the reader knows which policy scored.
    assert _says(result.output, "dial: 0.25 balanced (default)")
    assert "policy:knn(guarded,fallback" in flat


def test_an_anchor_outside_the_pool_is_refused_before_anything_is_spent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A flag error must not surface after the sweep has been paid for.

    The anchor has to name a pool model, and the pool is fully loaded by the pre-flight, so this
    is knowable for free. It used to be caught only in the report stage, by which point the sweep
    was bought, the policy fitted, and the dial set.
    """
    world_model = _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9})
    root = _project(tmp_path)
    result = _run(tmp_path, root, "--yes", "--baseline", "ghost")
    assert result.exit_code != 0
    assert _says(result.output, "--baseline 'ghost' is not a model in")
    assert _says(result.output, "Available: cheap, pricey")
    assert world_model.tasks == []  # not one cell bought
    assert not _paths(root)[0].exists()


def test_a_missing_world_model_names_the_command_a_user_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_seams(monkeypatch)
    root = _project(tmp_path)
    result = runner.invoke(
        app,
        ["optimize", "model", "ghost", "--root", str(root), "--pool", str(_pool_file(tmp_path))],
    )
    assert result.exit_code != 0
    assert "ghost" in _flat(result.output)


def test_a_refit_retires_the_stale_dial_snapshot_it_superseded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A chained refit-then-dial must not trip `route tune`'s stale-snapshot refusal.

    That refusal exists for a human who refits by hand and would otherwise dial the superseded
    fit back over the new one. Here the refit and the dial are the same command's own consecutive
    stages, so the snapshot is stale by construction and is retired explicitly.
    """
    _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9})
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes").exit_code == 0
    _matrix, policy_path, _report, _manifest = _paths(root)
    base_path = policy_path.parent / "policy.base.json"
    first_fit = RoutingPolicy.load(base_path).fitted_from

    forced = _run(tmp_path, root, "--yes", "--force-from", "sweep")
    assert forced.exit_code == 0, forced.output
    assert _says(forced.output, "re-baselined the dial")
    # The snapshot is the NEW fit as fitted, and the served policy is the new fit dialed.
    assert base_path.is_file()
    assert RoutingPolicy.load(base_path).fitted_from != first_fit
    assert RoutingPolicy.load(base_path).cost_quality is None
    assert RoutingPolicy.load(policy_path).cost_quality == 0.25


def test_a_redo_that_reproduces_the_same_fit_keeps_its_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The negative control: only a SUPERSEDED snapshot is retired, never a matching one."""
    _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9})
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes").exit_code == 0
    _matrix, policy_path, _report, _manifest = _paths(root)
    base_path = policy_path.parent / "policy.base.json"
    before = base_path.read_bytes()

    # --force-from fit refits the SAME matrix, so the fit (and its provenance) is identical.
    forced = _run(tmp_path, root, "--yes", "--force-from", "fit")
    assert forced.exit_code == 0, forced.output
    assert not _says(forced.output, "re-baselined the dial")
    assert base_path.read_bytes() == before


def test_the_dial_setting_reaches_the_served_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--cost-quality is the one operating-point decision, and it lands on the artifact."""
    _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9})
    root = _project(tmp_path)
    result = _run(tmp_path, root, "--yes", "--cost-quality", "0")
    assert result.exit_code == 0, result.output
    policy = RoutingPolicy.load(_paths(root)[1])
    assert policy.cost_quality == 0.0
    assert policy.floor_q == 0.5  # the quality end of the measured frontier
    assert _says(result.output, "dial: 0 quality max")


def test_the_sweep_persists_the_world_models_own_spend_beside_the_build_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The eval-infrastructure half of a sweep's bill has to land somewhere accountable.

    `route sweep` has always said the simulator's cost is "metered separately" while nothing
    persisted it: the world model opens one metered session per episode and every record died
    with its env. They are rolled into one `kind="sweep"` record in the model's own runs dir,
    beside the build and serve records already there.
    """
    _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9}, session_usd=0.02)
    root = _project(tmp_path)
    result = _run(tmp_path, root, "--yes")
    assert result.exit_code == 0, result.output

    runs = load_runs(root / "models" / "support" / "runs")
    sweeps = [record for record in runs if record.kind == "sweep"]
    assert len(sweeps) == 1
    swept = sweeps[0]
    # 6 episodes x $0.02, kept split by phase so the serve half and the judge half stay separable.
    assert swept.total.cost_usd == pytest.approx(0.12)
    assert swept.by_phase[Phase.SERVE].cost_usd == pytest.approx(0.09)
    assert swept.by_phase[Phase.JUDGE].cost_usd == pytest.approx(0.03)
    assert swept.total.calls == 18  # 6 sessions x (2 serve + 1 judge)


def test_the_two_sides_of_the_bill_are_reported_but_never_blended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Candidate spend is what serving costs; world-model spend is what measuring costs.

    One number covering both would overstate the price of serving the policy and understate the
    price of producing the evidence, so they are printed as two labeled lines and stored as two
    fields.
    """
    _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9}, session_usd=0.02)
    root = _project(tmp_path)
    result = _run(tmp_path, root, "--yes")
    assert result.exit_code == 0, result.output
    assert _says(result.output, "measured candidate spend $0.0013")
    assert _says(result.output, "measured world-model spend $0.1200 over 6 session(s)")
    assert _says(result.output, "eval infrastructure, not serving cost")
    assert _says(result.output, 'recorded as kind="sweep"')

    _matrix, _policy, _report, manifest_path = _paths(root)
    manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    sweep = manifest.record_for(Stage.SWEEP)
    assert sweep is not None
    assert sweep.spend_usd == pytest.approx(0.00132)  # candidate side, unchanged
    assert sweep.world_model_spend_usd == pytest.approx(0.12)  # its own field, never summed in
    assert sweep.total_spend_usd == pytest.approx(0.12132)  # only the cap adds them


def test_the_spend_cap_counts_the_world_model_side_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cap is a question about money leaving the account, and both sides do.

    The candidate side of this sweep projects at $0.33 and measures at $0.0013; the world-model
    side measures $0.12. A cap of $0.50 clears the pre-sweep projection either way, so the only
    thing that can stop the SECOND, freshly-forced sweep is the first run's total, and it only
    exceeds $0.50 once the world-model side is counted.
    """
    world_model = _patch_seams(
        monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9}, session_usd=0.05
    )
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes", "--max-usd", "0.60").exit_code == 0
    after_first = len(world_model.tasks)
    # First run: $0.0013 candidate + $0.30 world model = $0.3013 recorded.
    stopped = _run(tmp_path, root, "--yes", "--max-usd", "0.60", "--force-from", "sweep")
    assert stopped.exit_code == 1, stopped.output
    assert len(world_model.tasks) == after_first  # not one new episode
    assert _says(stopped.output, "stopped at the spend cap")
    assert _says(stopped.output, "$0.30 of its $0.60 cap")


def test_a_candidate_only_cap_would_have_let_that_second_sweep_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The negative control: with no world-model spend to count, the same cap does not trip."""
    world_model = _patch_seams(
        monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9}, session_usd=0.0
    )
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes", "--max-usd", "0.60").exit_code == 0
    after_first = len(world_model.tasks)
    again = _run(tmp_path, root, "--yes", "--max-usd", "0.60", "--force-from", "sweep")
    assert again.exit_code == 0, again.output
    assert len(world_model.tasks) > after_first


def test_the_first_sweep_says_the_world_model_side_is_not_projectable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Before a model's first sweep there is nothing to forecast from, and silence would mislead.

    "Not in this figure" reads as "not much" unless the line says how large that side can get:
    measured 7.0x the candidate side on a real tau corpus. So the caveat states both that it is
    unprojectable and that the printed total is a lower bound.
    """
    _patch_seams(monkeypatch)
    root = _project(tmp_path)
    answer = _Answer(False)
    monkeypatch.setattr(optimize_module, "Confirm", answer)
    monkeypatch.setattr(optimize_module, "_console", Console(width=240, force_terminal=True))
    result = _run(tmp_path, root)
    assert result.exit_code == 0, result.output
    assert _says(result.output, "not projectable before this model's first sweep")
    assert _says(result.output, "7.0x the candidate side")
    assert _says(result.output, "treat the number above as a lower bound")
    # No forecast is invented, and the run still proceeds to the confirmation.
    assert "projected~$" not in _flat(result.output)
    assert answer.asked


def test_a_second_sweep_forecasts_the_world_model_side_from_the_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once this model has been swept, its OWN measured ratio is the honest basis for a forecast."""
    _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9}, session_usd=0.02)
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes").exit_code == 0
    # First sweep: $0.00132 candidate, $0.12 world model, a ratio of ~90.9x.
    forced = _run(tmp_path, root, "--yes", "--force-from", "sweep")
    assert forced.exit_code == 0, forced.output
    assert _says(forced.output, "plus a projected ~$30.00 world-model side")
    assert _says(
        forced.output, "measured $0.1200 world-model against $0.0013 projectable candidate"
    )
    assert _says(forced.output, "90.9x")
    assert _says(forced.output, "a forecast from one prior sweep, not arithmetic")


def test_the_forecast_stops_a_sweep_a_candidate_only_cap_would_have_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of the allowance: the cap sees the money before it is spent, not after.

    The candidate projection alone is $0.33, so a $5 cap clears it easily and the sweep would
    start. The first sweep measured a 90.9x world-model ratio, which forecasts ~$30 for the same
    grid, so the run stops BEFORE buying any of it and says what the forecast rests on.
    """
    world_model = _patch_seams(
        monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9}, session_usd=0.02
    )
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes").exit_code == 0
    after_first = len(world_model.tasks)

    stopped = _run(tmp_path, root, "--yes", "--max-usd", "5", "--force-from", "sweep")
    assert stopped.exit_code == 1, stopped.output
    assert len(world_model.tasks) == after_first  # not one episode bought
    assert _says(stopped.output, "stopped at the spend cap")
    assert _says(stopped.output, "projection basis: the last sweep of this model measured")
    assert _says(stopped.output, "wmo optimize model support --max-usd <more>")


def test_without_the_forecast_that_same_cap_would_not_have_stopped_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The negative control: with no world-model spend to learn a ratio from, $5 clears $0.33."""
    world_model = _patch_seams(
        monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9}, session_usd=0.0
    )
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes").exit_code == 0
    after_first = len(world_model.tasks)
    again = _run(tmp_path, root, "--yes", "--max-usd", "5", "--force-from", "sweep")
    assert again.exit_code == 0, again.output
    assert len(world_model.tasks) > after_first


def test_a_sweep_the_coverage_contract_rejects_is_recorded_and_not_re_bought(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rejected sweep still cost money, so resume has to preserve it.

    `pricey` is throttled on every call, so its cells go unscored while `cheap` is scored on all
    three scenarios: the two would be ranked on different task sets, and the contract withholds
    the fit. The cells were paid for either way, and the printed message promises re-running will
    not buy them again. It has to be true.
    """
    world_model = _patch_seams(
        monkeypatch, rewards={"cheap-1": 0.4}, throttled_models=frozenset({"pricey-1"})
    )
    root = _project(tmp_path)
    rejected = _run(tmp_path, root, "--yes")
    assert rejected.exit_code == 1, rejected.output
    matrix_path, policy_path, _report, manifest_path = _paths(root)
    assert matrix_path.is_file() and not policy_path.exists()
    assert _says(rejected.output, "DIFFERENT scenarios")
    assert _says(rejected.output, "will not buy these cells again")
    bought = len(world_model.tasks)

    manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    sweep = manifest.record_for(Stage.SWEEP)
    assert sweep is not None and sweep.spend_usd > 0.0

    # The documented way forward: accept the bias. It must skip the sweep, not repeat it.
    accepted = _run(tmp_path, root, "--yes", "--allow-uneven-coverage")
    assert accepted.exit_code == 0, accepted.output
    assert len(world_model.tasks) == bought  # not one cell re-bought
    assert _says(accepted.output, "matrix.json is current")
    assert _says(accepted.output, "bias accepted")
    assert policy_path.is_file()  # and the fit finally happened


def test_the_coverage_contract_still_binds_when_the_sweep_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recording a rejected sweep must not become a way to smuggle a biased matrix into a fit.

    This is the hole that opens if the contract is enforced at the end of the sweep instead of in
    front of the fit: the sweep is recorded, the next run skips it, and nothing re-checks the
    evidence. So the gate lives with the fit and binds on the skip path too.
    """
    world_model = _patch_seams(
        monkeypatch, rewards={"cheap-1": 0.4}, throttled_models=frozenset({"pricey-1"})
    )
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes").exit_code == 1
    bought = len(world_model.tasks)
    _matrix, policy_path, _report, _manifest = _paths(root)

    # Same inputs, no --allow-uneven-coverage: the sweep is skipped (so nothing is spent) and the
    # fit is STILL withheld rather than quietly proceeding on the biased matrix.
    again = _run(tmp_path, root, "--yes")
    assert again.exit_code == 1, again.output
    assert len(world_model.tasks) == bought
    assert _says(again.output, "matrix.json is current")  # the sweep really was skipped
    assert _says(again.output, "DIFFERENT scenarios")  # and the gate still ran
    assert not policy_path.exists()


def test_accepting_biased_evidence_does_not_stick_silently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Consent to fit on uneven evidence is an input to the fit, so revoking it must be noticed.

    Without `allow_uneven` in the fit's fingerprint the flag is a one-way door: grant it once and
    every later run skips the fit, never reaches the coverage gate, and serves a policy fitted on
    knowingly-biased evidence with nothing in the output saying so.
    """
    _patch_seams(monkeypatch, rewards={"cheap-1": 0.4}, throttled_models=frozenset({"pricey-1"}))
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes").exit_code == 1  # withheld
    accepted = _run(tmp_path, root, "--yes", "--allow-uneven-coverage")
    assert accepted.exit_code == 0, accepted.output

    # Same project, flag withdrawn: the fit is re-planned because its inputs changed, the gate
    # runs again, and the bias is refused rather than silently inherited.
    revoked = _run(tmp_path, root, "--yes")
    assert revoked.exit_code == 1, revoked.output
    assert _says(revoked.output, "allow_uneven changed")
    assert _says(revoked.output, "DIFFERENT scenarios")


def test_the_cap_refuses_before_asking_rather_than_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Being asked to approve a run and then told it cannot start is the wrong order."""
    world_model = _patch_seams(monkeypatch)
    root = _project(tmp_path)
    answer = _Answer(True)
    monkeypatch.setattr(optimize_module, "Confirm", answer)
    monkeypatch.setattr(optimize_module, "_console", Console(width=240, force_terminal=True))
    result = _run(tmp_path, root, "--max-usd", "0.01")
    assert result.exit_code == 1, result.output
    assert answer.asked == []  # never asked
    assert world_model.tasks == []
    assert _says(result.output, "stopped at the spend cap")


def test_a_zero_priced_pool_is_still_confirmed_because_the_simulator_is_not_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keying the question on the candidate projection skips it exactly when it matters most.

    A pool priced at zero projects $0.00 candidate-side, but the sweep still spends on the world
    model. That is the case where the simulator's cost IS the whole bill, so the question has to
    key on "will this run buy cells", not on a candidate-side number.
    """
    world_model = _patch_seams(monkeypatch, session_usd=0.05)
    root = _project(tmp_path)
    free_pool = tmp_path / "free-pool.toml"
    free_pool.write_text(
        "[[model]]\n"
        'name = "cheap"\n'
        'kind = "openai"\n'
        'model = "cheap-1"\n'
        "input_per_mtok = 0.0\n"
        "output_per_mtok = 0.0\n",
        encoding="utf-8",
    )
    answer = _Answer(False)
    monkeypatch.setattr(optimize_module, "Confirm", answer)
    monkeypatch.setattr(optimize_module, "_console", Console(width=240, force_terminal=True))
    result = _run(tmp_path, root, pool=free_pool)
    assert result.exit_code == 0, result.output
    assert len(answer.asked) == 1  # asked, despite a $0.00 candidate projection
    assert world_model.tasks == []  # and declining bought nothing


def test_the_cap_counts_spend_from_runs_whose_records_were_superseded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--max-usd` bounds the optimization, and a re-swept stage's money still left the account.

    The manifest keeps only the LATEST record per stage, so seeding the cap from the stage rows
    forgets every superseded sweep. Three sweeps of $0.30 must read as $0.90 spent, not $0.30.
    """
    _patch_seams(monkeypatch, rewards={"cheap-1": 0.4, "pricey-1": 0.9}, session_usd=0.05)
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes").exit_code == 0
    assert _run(tmp_path, root, "--yes", "--force-from", "sweep").exit_code == 0

    _matrix, _policy, _report, manifest_path = _paths(root)
    manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    sweep = manifest.record_for(Stage.SWEEP)
    assert sweep is not None
    # One sweep's record survives, but both sweeps' spend does.
    assert manifest.lifetime_spend_usd == pytest.approx(sweep.total_spend_usd * 2, rel=1e-6)
