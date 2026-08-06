#!/usr/bin/env python3
"""Convert Kimi-K2.6 computer-use trajectories into the wmh OTel-GenAI trace corpus.

The source is real macOS GUI-control agent runs: an LLM agent drives the desktop by reading the
Accessibility tree and issuing `bash`/`osascript`/`read` tool calls, and the REAL tool output is
recorded per call — AX-tree dumps, command stdout, `read` file contents, and real failures. That
maps directly to the harness contract: one Step per tool call, with the real
`(action) -> observation` the agent actually saw. The environment being reconstructed is a macOS
desktop under GUI automation: predict the tool's real output given the tool call.

This is a stdlib-only converter (no `wmh` import, no third-party deps) so it stays a self-contained
capture tool. It reads the SOURCE JSONL in place and STREAMS it line by line — each line is one
trajectory whose `events` field is ~97% of the bytes (a per-frame render log we never need). We
json.loads the whole line but touch only the top-level scalars and `steps`; `events`/`messages`/
top-level `tool_calls` are ignored. Only the produced OTel JSONL is written to ``--out``.

Per trajectory, per `steps[].tool_calls[]` entry (in step order, then call order within a step):
  - action      = the real tool call (name + arguments, e.g. bash {"command": "..."}, read {...}).
  - observation = the real recorded `output`, with `is_error` from the call's `isError` flag.
  - task        = the trajectory's task instruction (carried on the first step as gen_ai.prompt).
  - Trace.metadata = benchmark, task_category, task_url, trajectory_id, model.
Steps with no tool call (final-answer turns) are dropped: open-loop replay scores predicted
observations for `(state, action)`, and a turn with no tool call has no environment observation.

`state_before` is left empty: a live desktop has no compact, non-leaky state snapshot to feed, and
open-loop replay reconstructs from the action + retrieved similar steps + teacher-forced history
(same rationale as the tau/terminal converters).

SANITIZATION (hard rule): every emitted string (task, arguments, output) is rewritten and the
finished file is ASSERTED clean before exiting (`corpus_test.py` guards the same invariants):
  - Internal path token: the source is laced with `screenpipe/synth` (skill paths, commands, `read`
    outputs). The token `screenpipe` must not survive. `screenpipe/synth` -> `agent` (so
    `/Users/m1/screenpipe/synth/...` -> `/Users/m1/agent/...`, `~/screenpipe/synth/...` ->
    `~/agent/...`), with a bare-`screenpipe` catch-all.
  - Credential-shaped tokens: AX-tree reads of API docs capture live-looking secrets (e.g. a
    Stripe `sk_test_...` key on stripe.com's docs — GitHub push protection rejects these). The
    secret body is redacted to `<prefix>_REDACTED`, keeping the informative prefix (which kind of
    key) while removing the secret and yielding a string no scanner flags.

Expected source schema (one trajectory per JSONL line):
  {"trajectory_id": "kimi_0001", "task": "...", "task_category": "...", "task_url": "...",
   "model": "Kimi-K2.6", "steps": [{"cot": "...", "text": "...", "stop_reason": "...",
     "tool_calls": [{"id": "...", "name": "read", "arguments": {...}, "output": "...",
       "isError": false}]}], "events": [...], ...}

Usage:
    python convert_to_wmh.py <trajectories.jsonl> --out traces.otel.jsonl --benchmark gui-tasks
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

# The banned internal token and the path segment it lives in. `screenpipe/synth` collapses to a
# single neutral segment so `/Users/m1/screenpipe/synth/skills/...` -> `/Users/m1/agent/skills/...`
# and `~/screenpipe/synth/...` -> `~/agent/...` uniformly; the bare catch-all removes any straggler.
BANNED_TOKEN = "screenpipe"

# Credential-shaped tokens captured verbatim from AX-tree reads of API docs (e.g. Stripe's own
# `sk_test_...` example key on stripe.com). The secret body is replaced but the prefix is kept, so
# the redaction is self-describing and the result matches no secret scanner. The trailing `_R...`
# has an underscore, which these key formats never contain, guaranteeing it can't be mistaken for a
# real key.
_SECRET_PATTERN = re.compile(r"\b((?:sk|rk|pk)_(?:live|test))_[A-Za-z0-9]{10,}")

_SANITIZE_RULES = (
    (re.compile(r"screenpipe/synth", re.IGNORECASE), "agent"),
    (re.compile(r"screenpipe", re.IGNORECASE), "agent"),
    (_SECRET_PATTERN, r"\1_REDACTED"),
)


def _sanitize(text: str) -> str:
    """Strip the internal path token and redact credential-shaped tokens (see module docstring)."""
    for pattern, replacement in _SANITIZE_RULES:
        text = pattern.sub(replacement, text)
    return text


def _attr(key: str, value: str) -> dict[str, Any]:
    return {"key": key, "value": {"stringValue": value}}


def _as_text(value: Any) -> str:  # noqa: ANN401 - tool output is loosely-typed JSON
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _trace_id(benchmark: str, trajectory_id: str) -> str:
    return hashlib.sha256(f"{benchmark}|{trajectory_id}".encode()).hexdigest()[:32]


def _metadata(traj: dict[str, Any], benchmark: str) -> dict[str, Any]:
    return {
        "benchmark": benchmark,
        "task_category": _sanitize(_as_text(traj.get("task_category", ""))),
        "task_url": _sanitize(_as_text(traj.get("task_url", ""))),
        "trajectory_id": _as_text(traj.get("trajectory_id", "")),
        "model": "gui-agent",
    }


def _iter_tool_calls(traj: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten `steps[].tool_calls[]` into one ordered list (step order, then call order)."""
    calls: list[dict[str, Any]] = []
    for step in traj.get("steps") or []:
        if not isinstance(step, dict):
            continue
        for tc in step.get("tool_calls") or []:
            if isinstance(tc, dict):
                calls.append(tc)
    return calls


def _spans_for_trajectory(
    traj: dict[str, Any], *, benchmark: str, trace_id: str
) -> list[dict[str, Any]]:
    """Emit ordered action/observation span pairs for one trajectory's tool calls."""
    raw_task = traj.get("task") if isinstance(traj.get("task"), str) else ""
    task_text = _sanitize(raw_task)
    metadata = _metadata(traj, benchmark)

    spans: list[dict[str, Any]] = []
    for ordinal, tc in enumerate(_iter_tool_calls(traj)):
        name = tc.get("name", "bash")
        args = tc.get("arguments") or {}
        args_json = _sanitize(json.dumps(args))
        obs_content = _sanitize(_as_text(tc.get("output")))
        obs_error = bool(tc.get("isError", False))

        action_attrs = [
            _attr("gen_ai.operation.name", "chat"),
            _attr("gen_ai.request.model", "gui-agent"),
            _attr("gen_ai.tool.name", str(name)),
            _attr("gen_ai.tool.call.arguments", args_json),
        ]
        if ordinal == 0 and task_text:
            action_attrs.append(_attr("gen_ai.prompt", task_text))
        if ordinal == 0:
            action_attrs.append(_attr("wmh.trace.metadata", json.dumps(metadata)))

        spans.append({
            "traceId": trace_id,
            "spanId": f"{trace_id[:12]}{ordinal:04x}a",
            "parentSpanId": "",
            "name": "chat gui",
            "startTimeUnixNano": ordinal * 10,
            "endTimeUnixNano": ordinal * 10 + 1,
            "status": {"code": "STATUS_CODE_OK"},
            "attributes": action_attrs,
        })
        spans.append({
            "traceId": trace_id,
            "spanId": f"{trace_id[:12]}{ordinal:04x}b",
            "parentSpanId": "",
            "name": "execute_tool gui",
            "startTimeUnixNano": ordinal * 10 + 2,
            "endTimeUnixNano": ordinal * 10 + 3,
            "status": {"code": "STATUS_CODE_ERROR" if obs_error else "STATUS_CODE_OK"},
            "attributes": [
                _attr("gen_ai.operation.name", "execute_tool"),
                _attr("gen_ai.tool.name", str(name)),
                _attr("gen_ai.tool.message", obs_content),
            ],
        })
    return spans


def _assert_clean(path: Path) -> None:
    """Fail loudly if the banned token or a credential-shaped token survived in the finished output.

    Both are hard sanitization invariants: the banned token is a case-insensitive raw-bytes check,
    and the secret check reuses the exact pattern the redaction targets (a surviving match means the
    redaction missed a variant — extend `_SANITIZE_RULES` before committing the corpus).
    """
    needle = BANNED_TOKEN.encode()
    with path.open("rb") as f:
        for i, raw in enumerate(f):
            if needle in raw.lower():
                raise AssertionError(
                    f"banned token {BANNED_TOKEN!r} survived sanitization at {path} line {i + 1}; "
                    "extend _SANITIZE_RULES to cover this variant before committing the corpus"
                )
            if _SECRET_PATTERN.search(raw.decode("utf-8", "ignore")):
                raise AssertionError(
                    f"credential-shaped token survived sanitization at {path} line {i + 1}; "
                    "extend _SANITIZE_RULES to cover this variant before committing the corpus"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", help="Path to the Kimi trajectories JSONL (read in place, streamed line by line)"
    )
    parser.add_argument("--out", required=True, help="Output OTel JSONL path")
    parser.add_argument("--benchmark", default="gui-tasks", help="Benchmark name")
    args = parser.parse_args()

    out_path = Path(args.out)
    n_traces = n_spans = n_skipped = 0
    with out_path.open("w", encoding="utf-8") as out, Path(args.source).open(
        encoding="utf-8"
    ) as src:
        for line in src:
            line = line.strip()
            if not line:
                continue
            traj = json.loads(line)
            if not isinstance(traj, dict):
                continue
            trajectory_id = _as_text(traj.get("trajectory_id", str(n_traces)))
            trace_id = _trace_id(args.benchmark, trajectory_id)
            spans = _spans_for_trajectory(traj, benchmark=args.benchmark, trace_id=trace_id)
            if not spans:
                n_skipped += 1
                continue
            for span in spans:
                out.write(json.dumps(span) + "\n")
                n_spans += 1
            n_traces += 1

    _assert_clean(out_path)
    print(
        f"wrote {n_traces} traces, {n_spans} spans -> {out_path} "
        f"(skipped {n_skipped} tool-call-less trajectories); "
        f"verified zero {BANNED_TOKEN!r} occurrences and no surviving credential-shaped tokens"
    )


if __name__ == "__main__":
    main()
