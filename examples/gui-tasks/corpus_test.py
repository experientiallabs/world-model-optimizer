"""Guard test for the committed gui-tasks corpus.

Locks the corpus invariants a regeneration must preserve: the trace/step counts, the hard
zero-`screenpipe` sanitization rule (checked over raw bytes AND the sanitizer unit), and that the
corpus loads through wmh's real OTel-GenAI ingest into tool-call Steps carrying their task.

`examples/` is excluded from the root ruff/ty/pytest gate (it holds self-contained, benchmark-
specific capture helpers), so this runs explicitly: `uv run pytest examples/gui-tasks/corpus_test.py`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from wmh.ingest import get_adapter

_DIR = Path(__file__).resolve().parent
_CORPUS = _DIR / "traces.otel.jsonl"

EXPECTED_TRACES = 1000
EXPECTED_STEPS = 16659


def _load_converter():
    """Import the sibling stdlib converter (examples/ is off the import path)."""
    spec = importlib.util.spec_from_file_location("gui_convert", _DIR / "convert_to_wmh.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_no_banned_token_in_corpus() -> None:
    """The banned internal token must not appear anywhere in the committed corpus (case-insensitive)."""
    with _CORPUS.open("rb") as f:
        for i, raw in enumerate(f):
            assert b"screenpipe" not in raw.lower(), f"banned token on line {i + 1}"


def test_sanitizer_rewrites_every_path_form() -> None:
    conv = _load_converter()
    assert conv._sanitize("/Users/m1/screenpipe/synth/skills/x") == "/Users/m1/agent/skills/x"
    assert conv._sanitize("~/screenpipe/synth/skills/x") == "~/agent/skills/x"
    assert "screenpipe" not in conv._sanitize("bare Screenpipe SCREENPIPE mention").lower()


def test_sanitizer_redacts_credential_tokens() -> None:
    conv = _load_converter()
    assert conv._sanitize("curl -u sk_test_51TmyBCGY5abcDEF123456: ...") == (
        "curl -u sk_test_REDACTED: ..."
    )
    assert conv._sanitize("key pk_live_AbCd1234567890xyz done") == "key pk_live_REDACTED done"
    assert not conv._SECRET_PATTERN.search(conv._sanitize("sk_test_51TmyBCGY5abcDEF123456"))


def test_no_credential_tokens_in_corpus() -> None:
    """No credential-shaped token (Stripe sk/rk/pk key) survives in the committed corpus."""
    conv = _load_converter()
    with _CORPUS.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            assert not conv._SECRET_PATTERN.search(line), f"credential token on line {i + 1}"


def test_ingest_round_trip() -> None:
    """The corpus loads through wmh's real ingest into tool-call Steps with tasks + metadata."""
    traces = get_adapter("otel-genai").from_file(str(_CORPUS))
    assert len(traces) == EXPECTED_TRACES
    assert sum(len(t.steps) for t in traces) == EXPECTED_STEPS

    for trace in traces:
        assert trace.steps, "every trace has at least one step"
        assert trace.metadata.get("benchmark") == "gui-tasks"
        assert trace.metadata.get("model") == "gui-agent"
        assert trace.steps[0].task, "the first step carries the task instruction"
        for step in trace.steps:
            assert step.action.kind.value == "tool_call"
            assert step.action.name


def test_metadata_carries_source_identity() -> None:
    traces = get_adapter("otel-genai").from_file(str(_CORPUS))
    ids = {t.metadata.get("trajectory_id") for t in traces}
    assert len(ids) == EXPECTED_TRACES, "trajectory ids are unique per trace"
    assert all(t.metadata.get("task_category") for t in traces)
