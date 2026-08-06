# appworld — a CUA world model scored against a coded ground-truth oracle

**Status: README-first PR — design committed before the corpus/model so the benchmark shape can be
reviewed; capture + world model + eval land in follow-up PRs on this branch. Full plan:
[`.agents/docs/proposals/ff2-cua-appworld.md`](../../.agents/docs/proposals/ff2-cua-appworld.md).**

Computer-use agents need private, resettable environments to practice in. The fashionable answer
is a "neural OS" — a model hallucinating the next screen. Recent examples: Microsoft's CUWM
([arXiv 2602.17365](https://arxiv.org/abs/2602.17365)) predicts a textual state-transition
description then synthesizes the next screenshot; gWorld
([arXiv 2602.01576](https://arxiv.org/abs/2602.01576)) generates the next mobile UI as executable
web code. Both are evaluated by *plausibility* — visual/textual similarity, downstream policy
lift — because real desktop and mobile apps expose no inspectable ground-truth state to score
against.

This example inverts that. [AppWorld](https://appworld.dev) (ACL 2024 best resource paper,
Apache-2.0) is an environment that is **entirely code**: 9 apps / 457 APIs simulated in Python
over SQLite, with deterministic seeded resets, snapshot/restore, and typed programmatic access to
every backend record. That makes it a **ground-truth oracle**: an agent does real work in it, we
capture the traces, build a world model from them with the standard `wmh` loop
(ingest → build → serve → eval), and then score the world model's predictions *exactly* —
per API-call outcome and per state transition — instead of by plausibility. It's the benchmark a
neural OS cannot run, in the same spirit that Dockerless
([arXiv 2606.28436](https://arxiv.org/abs/2606.28436)) validates its environment-free verifier
against execution-based ground truth.

## What gets measured

Prior finding (world-zoo Z5): LLM world models are strong *surface* simulators but unreliable on
**state-dependent gates** — auth, validation, availability, permissions, completion rules. Gates
are where this benchmark bites:

- **Gate accuracy (headline, open-loop).** On teacher-forced replay, a coded deterministic scorer
  (no LLM) compares the world model's predicted API outcome to the oracle's: success-vs-error,
  error class (auth / validation / not-found / permission / business-rule), and for mutating
  calls, whether the predicted state effect matches the oracle's actual DB diff. Reported per
  gate family.
- **Task-success agreement (closed-loop).** The same agent runs free against the oracle and
  against the world model from identical resets; AppWorld's own state-based `evaluate()` scores
  the oracle run, and we measure whether the world model told the same story — plus
  steps-until-first-gate-mismatch.
- **Open-loop fidelity** (`wmh eval`, D12 conventions) for continuity with the tau-bench /
  terminal-tasks / swe-bench numbers.

## Observations

Text-first, deliberately: actions are API calls, observations are the oracle's real JSON
responses, and state snapshots ride `wmh.state.structured`. CUWM's own factorization — predict
*what changes* textually, then render *how it appears* — argues state correctness is the layer
that matters; pixels are a presentation layer over the same state. Images are the stated future
direction, alongside a second, more visual benchmark leg (see the proposal).

## Layout (when complete)

```text
examples/appworld/
  README.md            # this file
  run.sh               # example-local venv bootstrap (appworld + its data; gate-excluded)
  capture_appworld.py  # agent (Bedrock Opus, sharded) solves train/dev tasks in the real env
  convert_to_wmh.py    # stdlib-only: episode records -> traces.otel.jsonl
  gate_scorer.py       # deterministic gate-outcome extraction + comparison (no LLM)
  traces.otel.jsonl    # the committed corpus (train/dev tasks only; test GT is encrypted upstream)
  evals/default.toml   # open-loop fidelity suite
  models/appworld/     # prebuilt world model artifact
```

AppWorld and its data live in the example-local venv (`run.sh`), never in the repo gate.
