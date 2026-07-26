"""End-to-end integration test against a real Bedrock model.

Skipped unless ``AWS_REGION`` is set (same gate as the provider live smoke tests). It exercises the
full payoff path on a self-contained, tau2-shaped OTel trace (built inline below, no external
corpus): convert -> build (GEPA on a live LLM) -> persist -> load -> step. It writes into a tmp dir
and uses a small GEPA budget to stay cheap.
"""

from __future__ import annotations

import json
import os

import pytest

from wmh.config import ArtifactPaths, HarnessConfig
from wmh.core.types import Action, ActionKind
from wmh.engine.build import build
from wmh.engine.world_model import WorldModel
from wmh.providers import ProviderConfig, ProviderKind, get_provider

_MODEL = "us.anthropic.claude-opus-4-8"

# A minimal tau2-shaped OTel trace (one tool-call step) so the test is self-contained.
_SPANS = [
    {
        "traceId": "f" * 32,
        "spanId": "s1",
        "name": "chat",
        "startTimeUnixNano": 1,
        "attributes": [
            {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
            {"key": "gen_ai.tool.name", "value": {"stringValue": "bash"}},
            {
                "key": "gen_ai.tool.call.arguments",
                "value": {"stringValue": '{"command": "get_user u_kath"}'},
            },
            {"key": "gen_ai.prompt", "value": {"stringValue": "Look up user u_kath."}},
        ],
    },
    {
        "traceId": "f" * 32,
        "spanId": "s2",
        "name": "execute_tool",
        "startTimeUnixNano": 2,
        "attributes": [
            {"key": "gen_ai.operation.name", "value": {"stringValue": "execute_tool"}},
            {
                "key": "gen_ai.tool.message",
                "value": {"stringValue": '{"membership": "silver", "name": "Katherine Johnson"}'},
            },
        ],
    },
]


@pytest.mark.skipif(
    "AWS_REGION" not in os.environ,
    reason="no AWS_REGION; skipping live end-to-end build+step test",
)
def test_build_load_step_against_real_bedrock(tmp_path) -> None:  # noqa: ANN001 - pytest fixture; pragma: no cover - network
    try:
        _run_live_build_load_step(tmp_path)
    except Exception as exc:  # noqa: BLE001 - classify, re-raise below
        # Live test: Bedrock saturation is an environment condition, not a regression — skip
        # like we do for missing creds. Anything else is a real failure.
        if "ServiceUnavailable" in str(exc) or "Throttling" in str(exc):
            pytest.skip(f"Bedrock capacity-constrained right now: {exc}")
        raise


def _run_live_build_load_step(tmp_path) -> None:  # noqa: ANN001 - pytest tmp_path passthrough
    region = os.environ["AWS_REGION"]
    traces_file = tmp_path / "traces.jsonl"
    traces_file.write_text("\n".join(json.dumps(s) for s in _SPANS), encoding="utf-8")
    root = tmp_path / ".wmh"

    config = HarnessConfig(
        providers=[ProviderConfig(kind=ProviderKind.BEDROCK, model=_MODEL, region=region)],
        serve_provider=ProviderKind.BEDROCK,
        embed_dim=256,
        gepa_budget=3,
        train_split=0.5,
    )
    result = build(config, file=str(traces_file), root=str(root))

    # The build produced a non-empty winning prompt, a frontier, and a persisted artifact.
    assert result.prompt
    paths = ArtifactPaths(root)
    assert paths.optimized_prompt.read_text(encoding="utf-8")
    assert paths.index.exists()

    # Load the stored world model and step a real action; the env should answer coherently.
    provider = get_provider(ProviderConfig(kind=ProviderKind.BEDROCK, model=_MODEL, region=region))
    wm = WorldModel.load(str(root), provider)
    session = wm.new_session(task="Look up user u_kath.")
    obs = wm.step(
        session.id,
        Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": "get_user u_kath"}),
    )
    assert obs.content  # the model returned something
    assert len(session.history) == 1
