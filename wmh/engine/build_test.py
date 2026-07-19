"""Tests for the build pipeline (ingest -> split -> index -> GEPA -> persist), no network."""

from __future__ import annotations

import json

import pytest

from wmh.config import ArtifactPaths, HarnessConfig, verify_manifest
from wmh.core.types import Action, ActionKind, Observation, Step, Trace
from wmh.engine.build import build, split_traces, split_traces_3way
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmh.retrieval import HashingEmbedder


class FakeProvider:
    """Canned world-model JSON for rollouts; a fixed 'improved' prompt for GEPA reflection."""

    def __init__(self) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")
        self.systems: list[str] = []  # system prompt of every complete() call, for assertions

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        self.systems.append(system)
        if "improve the system prompt" in system:
            return Completion(text="IMPROVED ENV PROMPT")
        if "grade a world model" in system:  # the judge
            return Completion(
                text=(
                    '{"format": 0.5, "factuality": 0.5, "consistency": 0.5, '
                    '"realism": 0.5, "quality": 0.5, "critique": "be more specific"}'
                )
            )
        if "KNOWLEDGE BASE" in system:  # the knowledge-seeding extraction pass
            return Completion(
                text='{"rules": "- gate: lookups need a valid id", '
                '"entities": "- user u1", "schemas": "- get_user -> text"}'
            )
        return Completion(text='{"output": "ok", "is_error": false}')

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


def test_split_traces_is_deterministic_and_partitions() -> None:
    traces = [Trace(trace_id=f"t{i}") for i in range(50)]
    a_train, a_test = split_traces(traces, 0.8)
    b_train, b_test = split_traces(list(reversed(traces)), 0.8)
    # Same assignment regardless of order; every trace lands in exactly one side.
    assert {t.trace_id for t in a_train} == {t.trace_id for t in b_train}
    assert len(a_train) + len(a_test) == 50
    assert 0 < len(a_train) < 50  # roughly an 80/20 split, both sides non-empty


def test_split_traces_3way_partitions_disjointly_and_is_stable() -> None:
    traces = [Trace(trace_id=f"t{i}") for i in range(60)]
    train, val, test = split_traces_3way(traces, 0.6, 0.2)
    ids = lambda ts: {t.trace_id for t in ts}  # noqa: E731
    # Disjoint and exhaustive.
    assert ids(train) | ids(val) | ids(test) == ids(traces)
    assert (
        not (ids(train) & ids(val)) and not (ids(val) & ids(test)) and not (ids(train) & ids(test))
    )
    assert len(train) + len(val) + len(test) == 60
    assert all(len(s) > 0 for s in (train, val, test))
    # Order-independent (same stable hash).
    r_train, r_val, r_test = split_traces_3way(list(reversed(traces)), 0.6, 0.2)
    assert ids(train) == ids(r_train) and ids(val) == ids(r_val) and ids(test) == ids(r_test)


def test_split_traces_3way_train_is_prefix_compatible_with_2way() -> None:
    # The 3-way train set is exactly the 2-way train set at the same cut, so switching to a 3-way
    # split never reshuffles which traces GEPA trains on — it only carves val out of the old test.
    traces = [Trace(trace_id=f"t{i}") for i in range(60)]
    two_train, two_test = split_traces(traces, 0.6)
    three_train, three_val, three_test = split_traces_3way(traces, 0.6, 0.2)
    assert {t.trace_id for t in two_train} == {t.trace_id for t in three_train}
    assert {t.trace_id for t in two_test} == {t.trace_id for t in three_val} | {
        t.trace_id for t in three_test
    }


def test_split_traces_3way_rejects_degenerate_fractions() -> None:
    traces = [Trace(trace_id=f"t{i}") for i in range(10)]
    with pytest.raises(ValueError, match="train_frac"):
        split_traces_3way(traces, 0.7, 0.4)  # sums to > 1 -> empty test
    with pytest.raises(ValueError, match="train_frac"):
        split_traces_3way(traces, 0.0, 0.5)


def test_cap_gepa_valset_bounds_steps_and_keeps_at_least_one_trace() -> None:
    from wmh.engine.build import _GEPA_VAL_STEP_CAP, _cap_gepa_valset

    def trace_with_steps(tid: str, n: int) -> Trace:
        step = Step(
            action=Action(kind=ActionKind.TOOL_CALL, name="get", arguments={}),
            observation=Observation(content="ok"),
        )
        return Trace(trace_id=tid, steps=[step.model_copy() for _ in range(n)])

    many = [trace_with_steps(f"t{i}", 2) for i in range(200)]  # 400 steps uncapped
    capped = _cap_gepa_valset(many)
    assert sum(len(t.steps) for t in capped) <= _GEPA_VAL_STEP_CAP
    assert capped == many[: len(capped)]  # stable prefix of the (already shuffled) split

    # A single over-cap trace still passes through: never starve GEPA of a valset entirely.
    huge = trace_with_steps("huge", _GEPA_VAL_STEP_CAP + 10)
    assert _cap_gepa_valset([huge]) == [huge]


def _multi_trace_file(tmp_path, n: int) -> str:  # noqa: ANN001 - pytest fixture
    """Write `n` one-step OTel traces with distinct 32-char trace ids."""
    lines: list[str] = []
    for i in range(n):
        tid = f"{i:032d}"
        lines.append(
            json.dumps(
                {
                    "traceId": tid,
                    "spanId": "s1",
                    "name": "chat",
                    "startTimeUnixNano": 1,
                    "attributes": [
                        {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
                        {"key": "gen_ai.tool.name", "value": {"stringValue": "get_user"}},
                        {
                            "key": "gen_ai.tool.call.arguments",
                            "value": {"stringValue": '{"id": "u"}'},
                        },
                        {"key": "gen_ai.prompt", "value": {"stringValue": "look up u"}},
                    ],
                }
            )
        )
        lines.append(
            json.dumps(
                {
                    "traceId": tid,
                    "spanId": "s2",
                    "name": "execute_tool",
                    "startTimeUnixNano": 2,
                    "attributes": [
                        {"key": "gen_ai.operation.name", "value": {"stringValue": "execute_tool"}},
                        {"key": "gen_ai.tool.message", "value": {"stringValue": "found u"}},
                    ],
                }
            )
        )
    p = tmp_path / "traces.jsonl"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


def test_build_falls_back_to_base_when_gepa_prompt_is_empty(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    """An empty GEPA winner (weak reflection LM) must not be persisted; base is written instead."""
    import sys

    from wmh.engine.prompts import BASE_ENV_PROMPT
    from wmh.optimize import OptimizeResult

    build_mod = sys.modules["wmh.engine.build"]
    traces_file = _multi_trace_file(tmp_path, 12)

    class _EmptyOptimizer:
        def __init__(self, *a, **k) -> None:  # noqa: ANN002, ANN003
            pass

        def optimize(self, *a, **k) -> OptimizeResult:  # noqa: ANN002, ANN003
            return OptimizeResult(prompt="   \n", frontier=[])  # blank winner

    monkeypatch.setattr(build_mod, "GEPAOptimizer", _EmptyOptimizer)
    config = HarnessConfig(
        providers=[ProviderConfig(kind=ProviderKind.BEDROCK, model="m")],
        serve_provider=ProviderKind.BEDROCK,
        embed_dim=64,
        gepa_budget=2,
        train_split=0.7,
    )
    result = build(
        config,
        file=traces_file,
        root=str(tmp_path / ".wmh"),
        serve_provider=FakeProvider(),
        embedder=HashingEmbedder(dim=64),
    )
    assert result.prompt == BASE_ENV_PROMPT
    paths = ArtifactPaths(tmp_path / ".wmh")
    assert paths.optimized_prompt.read_text(encoding="utf-8").strip()  # never empty


def test_build_writes_a_loadable_artifact(tmp_path) -> None:  # noqa: ANN001 - pytest fixture
    # A tiny OTel JSONL with one tool-call step.
    span_llm = {
        "traceId": "a" * 32,
        "spanId": "s1",
        "name": "chat",
        "startTimeUnixNano": 1,
        "attributes": [
            {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
            {"key": "gen_ai.tool.name", "value": {"stringValue": "get_user"}},
            {"key": "gen_ai.tool.call.arguments", "value": {"stringValue": '{"id": "u1"}'}},
            {"key": "gen_ai.prompt", "value": {"stringValue": "look up u1"}},
        ],
    }
    span_tool = {
        "traceId": "a" * 32,
        "spanId": "s2",
        "name": "execute_tool",
        "startTimeUnixNano": 2,
        "attributes": [
            {"key": "gen_ai.operation.name", "value": {"stringValue": "execute_tool"}},
            {"key": "gen_ai.tool.message", "value": {"stringValue": "found u1"}},
        ],
    }
    traces_file = tmp_path / "traces.jsonl"
    traces_file.write_text(
        json.dumps(span_llm) + "\n" + json.dumps(span_tool) + "\n", encoding="utf-8"
    )

    root = tmp_path / ".wmh"
    config = HarnessConfig(
        providers=[ProviderConfig(kind=ProviderKind.BEDROCK, model="m")],
        serve_provider=ProviderKind.BEDROCK,
        embed_dim=64,
        gepa_budget=4,
        train_split=0.5,
    )
    result = build(
        config,
        file=str(traces_file),
        root=str(root),
        serve_provider=FakeProvider(),
        embedder=HashingEmbedder(dim=64),
    )

    paths = ArtifactPaths(root)
    assert paths.config.exists()
    assert paths.optimized_prompt.read_text(encoding="utf-8")  # a non-empty winning prompt
    assert json.loads(paths.frontier.read_text(encoding="utf-8"))  # frontier persisted
    assert result.prompt
    assert verify_manifest(root).ok
    # The index round-trips: a freshly loaded WorldModel can retrieve the indexed step.
    from wmh.engine.world_model import WorldModel

    wm = WorldModel.load(str(root), FakeProvider())
    assert wm.sample_steps(5)


def test_build_judge_stays_on_the_judge_provider(tmp_path) -> None:  # noqa: ANN001 - fixture
    # The serve provider may be a failover chain; the judge (GEPA's fitness metric) must run on
    # the separately supplied pinned provider so optimization is scored by one model throughout.
    span_llm = {
        "traceId": "b" * 32,
        "spanId": "s1",
        "name": "chat",
        "startTimeUnixNano": 1,
        "attributes": [
            {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
            {"key": "gen_ai.tool.name", "value": {"stringValue": "get_user"}},
            {"key": "gen_ai.tool.call.arguments", "value": {"stringValue": '{"id": "u1"}'}},
        ],
    }
    span_tool = {
        "traceId": "b" * 32,
        "spanId": "s2",
        "name": "execute_tool",
        "startTimeUnixNano": 2,
        "attributes": [
            {"key": "gen_ai.operation.name", "value": {"stringValue": "execute_tool"}},
            {"key": "gen_ai.tool.message", "value": {"stringValue": "found u1"}},
        ],
    }
    # Four copies of the trace under distinct trace ids: the 3-way split (train/val/test) must
    # leave a non-empty val set, or GEPA never scores a candidate and the judge is never called.
    lines: list[str] = []
    for i in range(4):
        trace_id = f"{i:x}" * 32
        lines.append(json.dumps({**span_llm, "traceId": trace_id, "spanId": f"s1-{i}"}))
        lines.append(json.dumps({**span_tool, "traceId": trace_id, "spanId": f"s2-{i}"}))
    traces_file = tmp_path / "traces.jsonl"
    traces_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    serve, judge = FakeProvider(), FakeProvider()
    config = HarnessConfig(
        providers=[ProviderConfig(kind=ProviderKind.BEDROCK, model="m")],
        serve_provider=ProviderKind.BEDROCK,
        embed_dim=64,
        gepa_budget=4,
        train_split=0.5,
    )
    result = build(
        config,
        file=str(traces_file),
        root=str(tmp_path / ".wmh"),
        serve_provider=serve,
        judge_provider=judge,
        embedder=HashingEmbedder(dim=64),
    )
    assert result.prompt
    assert all("grade a world model" not in s for s in serve.systems)
    assert any("grade a world model" in s for s in judge.systems)
    assert all("grade a world model" in s for s in judge.systems)  # judge does ONLY judging


def _tiny_trace_file(tmp_path) -> str:  # noqa: ANN001 - pytest fixture
    span_llm = {
        "traceId": "e" * 32,
        "spanId": "s1",
        "name": "chat",
        "startTimeUnixNano": 1,
        "attributes": [
            {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
            {"key": "gen_ai.tool.name", "value": {"stringValue": "get_user"}},
            {"key": "gen_ai.tool.call.arguments", "value": {"stringValue": '{"id": "u1"}'}},
            {"key": "gen_ai.prompt", "value": {"stringValue": "look up u1"}},
        ],
    }
    span_tool = {
        "traceId": "e" * 32,
        "spanId": "s2",
        "name": "execute_tool",
        "startTimeUnixNano": 2,
        "attributes": [
            {"key": "gen_ai.operation.name", "value": {"stringValue": "execute_tool"}},
            {"key": "gen_ai.tool.message", "value": {"stringValue": "found u1"}},
        ],
    }
    traces_file = tmp_path / "traces.jsonl"
    traces_file.write_text(
        json.dumps(span_llm) + "\n" + json.dumps(span_tool) + "\n", encoding="utf-8"
    )
    return str(traces_file)


def test_build_with_knowledge_seeds_the_kb_from_train(tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / ".wmh"
    config = HarnessConfig(
        providers=[ProviderConfig(kind=ProviderKind.BEDROCK, model="m")],
        serve_provider=ProviderKind.BEDROCK,
        embed_dim=64,
        gepa_budget=4,
        train_split=0.5,
        knowledge=True,
    )
    build(
        config,
        file=_tiny_trace_file(tmp_path),
        root=str(root),
        serve_provider=FakeProvider(),
        embedder=HashingEmbedder(dim=64),
    )
    paths = ArtifactPaths(root)
    rules = (paths.knowledge / "rules.md").read_text(encoding="utf-8")
    assert "gate: lookups need a valid id" in rules


class _AutoFidelityProvider(FakeProvider):
    """Reason-mode predictions are distinguishable so the judge can prefer them."""

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        user = messages[0].content
        if "improve the system prompt" in system:
            return Completion(text="IMPROVED ENV PROMPT")
        if "grade a world model" in system:  # rubric judge: reward the reason-mode prediction
            score = 0.9 if "ok-reason" in user else 0.4
            dims = ", ".join(
                f'"{d}": {score}'
                for d in ("format", "factuality", "consistency", "realism", "quality")
            )
            return Completion(text="{" + dims + ', "critique": "ok"}')
        if "KNOWLEDGE BASE" in system:
            return Completion(text='{"rules": "- gate: x", "entities": "", "schemas": ""}')
        if '"reasoning"' in user:  # reason-mode contract requested
            return Completion(text='{"reasoning": "r", "output": "ok-reason", "is_error": false}')
        return Completion(text='{"output": "ok", "is_error": false}')


def test_build_max_fidelity_finds_the_winner_but_leaves_defaults_plain(tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / ".wmh"
    config = HarnessConfig(
        providers=[ProviderConfig(kind=ProviderKind.BEDROCK, model="m")],
        serve_provider=ProviderKind.BEDROCK,
        embed_dim=64,
        gepa_budget=4,
        train_split=0.5,
    )
    build(
        config,
        file=_tiny_trace_file(tmp_path),
        root=str(root),
        serve_provider=_AutoFidelityProvider(),
        embedder=HashingEmbedder(dim=64),
        max_fidelity=True,
        fidelity_budget=2,
        full_search=True,
    )
    from wmh.config import load_config

    # The search NEVER changes the serve defaults — a plain run stays pure RAG...
    persisted = load_config(str(root))
    assert persisted.reasoning is False
    assert persisted.verify is False
    assert persisted.knowledge is False
    # ...the measured winner lives in the artifact as a runtime menu (--max-fidelity).
    report = json.loads((root / "auto_fidelity.json").read_text(encoding="utf-8"))
    assert report["winner_label"] == "reason"  # judge preferred reason; ties -> cheapest
    assert report["scores"]["base"] == pytest.approx(0.4)  # five-dim rubric mean
    # And loading with max_fidelity activates exactly the winner.
    from wmh.engine.world_model import WorldModel

    wm = WorldModel.load(str(root), _AutoFidelityProvider(), max_fidelity=True)
    session = wm.new_session(task="t")
    obs = wm.step(session.id, Action(kind=ActionKind.TOOL_CALL, name="get_user", arguments={}))
    assert obs.content == "ok-reason"


def test_build_low_tier_skips_gepa(tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / ".wmh"
    config = HarnessConfig(
        providers=[ProviderConfig(kind=ProviderKind.BEDROCK, model="m")],
        serve_provider=ProviderKind.BEDROCK,
        embed_dim=64,
        gepa_budget=0,  # the low fidelity tier: RAG only
        train_split=0.5,
    )

    class _NoGepaProvider(FakeProvider):
        def complete(self, system, messages, *, temperature=0.7, max_tokens=8192):  # noqa: ANN001, ANN202
            assert "improve the system prompt" not in system, "low tier must not run GEPA"
            return super().complete(
                system, messages, temperature=temperature, max_tokens=max_tokens
            )

    result = build(
        config,
        file=_tiny_trace_file(tmp_path),
        root=str(root),
        serve_provider=_NoGepaProvider(),
        embedder=HashingEmbedder(dim=64),
    )
    from wmh.engine.prompts import BASE_ENV_PROMPT

    assert result.prompt == BASE_ENV_PROMPT
    assert ArtifactPaths(root).optimized_prompt.read_text(encoding="utf-8") == BASE_ENV_PROMPT


def test_build_drop_degenerate_filters_all_empty_traces(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    # A capture can be polluted with all-empty-observation traces (swe-bench is ~66% such junk);
    # --drop-degenerate must filter them before the split so the model isn't built on damage.
    import importlib

    build_mod = importlib.import_module("wmh.engine.build")
    seen: dict[str, int] = {}
    real = build_mod.split_traces_3way

    def spy(traces, *a, **k):  # noqa: ANN001, ANN002, ANN003, ANN202
        seen["n"] = len(traces)
        return real(traces, *a, **k)

    monkeypatch.setattr(build_mod, "split_traces_3way", spy)
    config = HarnessConfig(
        providers=[ProviderConfig(kind=ProviderKind.BEDROCK, model="m")],
        serve_provider=ProviderKind.BEDROCK,
        embed_dim=64,
        gepa_budget=0,
        train_split=0.5,
    )
    # 4 real trace-pairs + 2 all-empty-observation degenerate traces appended.
    trace_file = _tiny_trace_file(tmp_path)
    import json as _json
    from pathlib import Path as _Path

    lines = _Path(trace_file).read_text(encoding="utf-8").splitlines()
    for i in range(2):
        tid = f"deadbeef{i:024x}"
        lines.append(_json.dumps({**_json.loads(lines[0]), "traceId": tid, "spanId": f"z{i}"}))
    _Path(trace_file).write_text("\n".join(lines) + "\n", encoding="utf-8")

    build(
        config,
        file=trace_file,
        root=str(tmp_path / ".wmh"),
        serve_provider=FakeProvider(),
        embedder=HashingEmbedder(dim=64),
        drop_degenerate=True,
    )
    # The degenerate single-span traces are gone before the split sees them.
    assert seen["n"] <= 4


def test_build_estimate_only_ships_the_signature_config_without_search(tmp_path) -> None:  # noqa: ANN001
    # The low tier: no GEPA, no LLM search, but a real winner (the signature estimate) is
    # persisted so --max-fidelity serves it and the higher tiers seed the same floor.
    import json as _json

    root = tmp_path / ".wmh"
    config = HarnessConfig(
        providers=[ProviderConfig(kind=ProviderKind.BEDROCK, model="m")],
        serve_provider=ProviderKind.BEDROCK,
        embed_dim=64,
        gepa_budget=0,
        train_split=0.5,
    )

    class _NoSearchProvider(FakeProvider):
        def complete(self, system, messages, *, temperature=0.7, max_tokens=8192):  # noqa: ANN001, ANN202
            assert "improve the system prompt" not in system, "estimate tier must not run GEPA"
            assert "grade a world model" not in system, "estimate tier must not score candidates"
            return super().complete(
                system, messages, temperature=temperature, max_tokens=max_tokens
            )

    build(
        config,
        file=_tiny_trace_file(tmp_path),
        root=str(root),
        serve_provider=_NoSearchProvider(),
        embedder=HashingEmbedder(dim=64),
        estimate_only=True,
    )
    report = _json.loads((root / "auto_fidelity.json").read_text(encoding="utf-8"))
    # The tiny fixture is a tool-call corpus -> the matrix estimate is `reason`.
    assert report["winner_label"] == "reason"
    assert report["winner_spec"]["reasoning"] is True
    assert report["scores"] == {}  # nothing was scored — pure estimate


def test_build_default_runs_no_search(tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / ".wmh"
    config = HarnessConfig(
        providers=[ProviderConfig(kind=ProviderKind.BEDROCK, model="m")],
        serve_provider=ProviderKind.BEDROCK,
        embed_dim=64,
        gepa_budget=4,
        train_split=0.5,
    )
    build(
        config,
        file=_tiny_trace_file(tmp_path),
        root=str(root),
        serve_provider=FakeProvider(),
        embedder=HashingEmbedder(dim=64),
    )
    assert not (root / "auto_fidelity.json").exists()  # default stays plain RAG, no search
