# gui-tasks — macOS computer-use corpus

A wmh trace corpus of **real macOS GUI-control agent runs**. A Kimi-K2.6 agent drives the desktop
by reading the Accessibility (AX) tree and issuing `bash` / `osascript` / `read` / `write` / `edit`
tool calls; the **real tool output** the agent saw is recorded per call. That is exactly the harness
contract — one Step per tool call, with the true `(action) -> observation` pair — so the world model
reconstructs a macOS desktop under GUI automation: **predict the tool's real output given the call.**

## What's here

| file | role |
|---|---|
| `convert_to_wmh.py` | self-contained, stdlib-only converter (source JSONL -> `traces.otel.jsonl`) |
| `traces.otel.jsonl` | the committed corpus: OTel-GenAI span JSONL that `wmh.ingest.otel_genai` reads |
| `corpus_test.py` | guard test: counts, zero banned token, ingest round-trip |

## Corpus stats

- **1000** traces (one per trajectory), **16 659** tool-call Steps (~16.7 steps/trace), **48.8 MB**.
- Every trajectory has ≥1 tool call, so none are dropped; final-answer turns with no tool call are
  dropped (no observation to predict).
- Action tool distribution: `bash` 15 314, `read` 1 163, `write` 167, `edit` 15.
- `task_category` distribution (verbatim from the source — the upstream labels are inconsistent,
  e.g. both `COMPARISON` and `Product research/comparison`; we preserve them rather than invent a
  taxonomy):

  ```
  COMPARISON 194 · DEVELOPER DOCS 158 · NEWS 117 · RESEARCH 91 · OPEN DATA 90 · GITHUB 89 ·
  MULTI-PAGE NAV 67 · EXTRACTION 57 · SHOPPING 37 · Product research/comparison 34 ·
  Shopping/food 18 · GitHub public repos 17 · Community/forums 16 ·
  Developer documentation/reference 9 · Maps/travel 6
  ```

## Conversion contract

One wmh Step per `steps[].tool_calls[]` entry, in step order then call order within a step
(multi-tool-call steps emit one Step each):

- **action** = the tool call — `name` + `arguments` (e.g. `bash {"command": ...}`, `read {"path": ...}`).
- **observation** = the recorded `output`, with `is_error` from the call's `isError` flag.
- **task** = the trajectory's instruction, carried on the first step as `gen_ai.prompt`.
- **`Trace.metadata`** = `{benchmark, task_category, task_url, trajectory_id, model: "gui-agent"}`.
- **`state_before`** left empty: a live desktop has no compact, non-leaky state snapshot to feed;
  open-loop replay reconstructs from the action + retrieved similar steps + teacher-forced history
  (same rationale as the tau-bench / terminal-tasks converters).

The span shape mirrors `../tau-bench/convert_to_wmh.py` and `../terminal-tasks/convert_to_wmh.py`
exactly (a `chat gui` action span + an `execute_tool gui` observation span per Step).

## Sanitization

The converter rewrites every emitted string — task, arguments, output — and **asserts** the finished
file is clean before exiting; `corpus_test.py` re-checks the same invariants:

- **Internal path token.** The upstream traces are laced with the segment `screenpipe/synth` (skill
  paths, commands, `read` outputs). The token **`screenpipe` must not appear anywhere.** Rewritten
  `screenpipe/synth -> agent` (so `/Users/m1/screenpipe/synth/...` -> `/Users/m1/agent/...`,
  `~/screenpipe/synth/...` -> `~/agent/...`) with a bare-`screenpipe` catch-all; asserted zero
  occurrences (case-insensitive).
- **Credential-shaped tokens.** AX-tree reads of API docs capture live-looking secrets — e.g. a
  Stripe `sk_test_...` example key on stripe.com's docs, which GitHub push protection rejects. The
  secret body is redacted to `<prefix>_REDACTED` (e.g. `sk_test_REDACTED`), keeping the informative
  prefix while removing the secret; asserted no `(sk|rk|pk)_(live|test)_...` token survives.

## Regenerate

The 9 GB source stays outside the repo and is read **in place, streamed line by line** (the `events`
field is ~97% of the bytes and is never used):

```bash
python examples/gui-tasks/convert_to_wmh.py \
  ~/Downloads/traces_kimi_k26_1000.jsonl \
  --out examples/gui-tasks/traces.otel.jsonl --benchmark gui-tasks
```

Then verify the corpus (not in the default `pytest` testpaths — `examples/` is excluded from the
root gate — so run it explicitly):

```bash
uv run pytest examples/gui-tasks/corpus_test.py -q
```
