"""Tests for segment-scoped compression (C2 round 3).

The protection guarantees are tested against the REAL rendered-turn format
(`wmo.env.llm_agent._render_turn`), so a format change breaks these tests loudly instead
of the scope silently protecting nothing.
"""

from __future__ import annotations

import wmo.optimize  # noqa: F401 - registers the scoped factories
from wmo.core.types import Action, ActionKind, EnvState, Observation, Step
from wmo.env.llm_agent import _render_turn
from wmo.optimize.compression import CompressionConfig, get_compressor
from wmo.optimize.compression_scoped import ScopedCompressor

CONFIG = CompressionConfig(
    compressor_id="scoped-truncate", compressor_version="1-scope1", aggressiveness=0.5
)

LONG_OBS = "line one of tool output\nline two with many words " + "filler " * 40
DIALOGUE_REPLY = "customer says: my order id is X-9931-B and I want a refund please"


def _step(kind: ActionKind, observation: str, *, error: bool = False) -> Step:
    action = (
        Action(kind=kind, name="search", arguments={"query": "orders"})
        if kind is ActionKind.TOOL_CALL
        else Action(kind=kind, content="what is your order id?")
    )
    return Step(
        action=action,
        observation=Observation(content=observation, is_error=error),
    )


def _turn(history: list[Step]) -> str:
    return _render_turn(
        "Refund order X-9931-B for the customer",
        EnvState(scratchpad="be polite"),
        history,
    )


def test_task_dialogue_and_actions_pass_through_byte_exact() -> None:
    turn = _turn(
        [
            _step(ActionKind.TOOL_CALL, LONG_OBS),
            _step(ActionKind.MESSAGE, DIALOGUE_REPLY),
            _step(ActionKind.TOOL_CALL, "short", error=True),
        ]
    )
    compressor = get_compressor("scoped-truncate")
    out = compressor.compress([turn], CONFIG).segments[0]

    assert out.startswith("TASK: Refund order X-9931-B for the customer")
    assert "ENVIRONMENT NOTES: be polite" in out
    assert DIALOGUE_REPLY in out  # dialogue observation untouched
    assert 'search({"query": "orders"})' in out  # tool syntax untouched
    assert out.rstrip().endswith("Your next move (JSON only):")
    assert " [ERROR]" in out  # error mark survives on the compressed entry
    assert "line one of tool output" in out  # head of the tool observation kept
    assert out.count("filler") < LONG_OBS.count("filler")  # the tail was compressed


def test_only_tool_observations_shrink() -> None:
    tool_turn = _turn([_step(ActionKind.TOOL_CALL, LONG_OBS)])
    dialogue_turn = _turn([_step(ActionKind.MESSAGE, LONG_OBS)])
    compressor = get_compressor("scoped-truncate")
    assert compressor.compress([tool_turn], CONFIG).segments[0] != tool_turn
    assert compressor.compress([dialogue_turn], CONFIG).segments[0] == dialogue_turn


def test_append_stability_across_growing_history() -> None:
    # Turn k+1 re-renders turn k's entries verbatim; the compressed bytes of the shared
    # entries must be identical (the cache-safety property the whole track rides on).
    history = [_step(ActionKind.TOOL_CALL, LONG_OBS), _step(ActionKind.MESSAGE, DIALOGUE_REPLY)]
    grown = [*history, _step(ActionKind.TOOL_CALL, "another tool result " + "word " * 30)]
    compressor = get_compressor("scoped-truncate")
    short = compressor.compress([_turn(history)], CONFIG).segments[0]
    longer = compressor.compress([_turn(grown)], CONFIG).segments[0]
    shared_prefix = short.rpartition("Your next move")[0]
    assert longer.startswith(shared_prefix)


def test_non_rendered_segments_pass_through_uncompressed() -> None:
    # Outside the eval path's format the scope cannot tell observations from dialogue;
    # the fail-safe direction is compressing nothing.
    raw = "just some text " * 50
    compressor = get_compressor("scoped-truncate")
    assert compressor.compress([raw], CONFIG).segments[0] == raw


def test_scope_rides_the_version_identity() -> None:
    compressor = get_compressor("scoped-truncate")
    assert compressor.id == "scoped-truncate"
    assert compressor.version == "1-scope1"  # a different scope rule = different bytes
    assert compressor.append_stable is True


def test_accounting_reflects_only_what_changed() -> None:
    turn = _turn([_step(ActionKind.TOOL_CALL, LONG_OBS)])
    compressor = ScopedCompressor(get_compressor("truncate"))
    result = compressor.compress([turn], CONFIG)
    assert result.tokens_in_compressed < result.tokens_in_raw
    assert result.segments[0].startswith("TASK:")
