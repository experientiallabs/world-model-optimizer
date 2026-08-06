---
area: Research
status: Proposal (user-interviewed 2026-07-07; design approved to plan, harness is a follow-up)
date: 2026-07-07
---

# A wmh-native world-model benchmark (working title: **WMH-Bench** — naming deferred)

AgentWorldBench (arXiv 2606.24597, "Qwen-AgentWorld") is the first external benchmark for
language world models: given an interaction history, predict the next observation, scored by an
LLM judge on five dimensions (Format, Factuality, Consistency, Realism, Quality). It defined the
category. It is also **judge-only** — every number it produces is a rubric opinion from one
pinned judge model.

wmh holds the inverse asset: **real captured corpora with deterministic graders**. Our benchmark
differentiates on exactly what a judge-only benchmark cannot do:

1. **Deterministic, judge-free signals** — reproducible by anyone, zero judge cost, no
   judge-version drift.
2. **Reward agreement under WM-replacement** — the same agent runs the same task against the
   real environment and against the world model, and the same deterministic grader scores both.
   No other benchmark can ask "does evaluating against the sim reach the same verdict as the
   real env?", because no other benchmark ships the real environments.

## User decisions (interview, 2026-07-07)

| Question | Decision |
|---|---|
| Scoring composition | **Deterministic core, empirically validated against the LLM judge.** The deterministic composite must match the pinned judge's performance; iterate until agreement is close. If it can't get close, fall back to a pinned LLM judge as headline. |
| Test split | **Public, SHA-pinned** (status quo: frozen test splits on HF, SHA-guarded). |
| Contamination canary | **Yes — design now, implement in the corpus-publishing follow-up** (not this PR). |
| Corpus scope | **All public corpora, licenses labeled.** appworld excluded (local-only license). CC BY-NC corpora (financebench, crmarena) included with the restriction disclosed on the card; gaia2's existing don't-train notice carried through. |
| Name/branding | **Deferred** — this doc uses "WMH-Bench" as a placeholder only. |
| Leaderboard | **None.** Static, version-pinned results table in `docs/` (and optionally the HF dataset cards). Consistent with WS-B1's no-leaderboard website decision. |

## The benchmark

One benchmark, two tracks. A submission is a *world model*: anything that maps
`(task, interaction history) → next observation` (and can hold session state).

### Track A — open-loop deterministic fidelity (all corpora)

Teacher-forced replay over the frozen test split (the existing `wmh eval` mechanic, D12
conventions): feed the real recorded `(state, action)` prefix, predict the observation, score
against the recorded observation. Deterministic per-step signals:

- **Error-flag accuracy** — did the WM predict `is_error` correctly? (Already reported today.)
- **Structured-field agreement** — benchmark-aware deterministic extractors over the observation
  text (exit codes, numeric answers, row counts, JSON keys — per-corpus extractor specs live
  with each adapter). This is the part that must be *built and validated*, see § Validation.

### Track B — reward agreement under WM-replacement (the differentiator)

The `wm_replace_demo` mechanic, generalized from the financebench prototype
(`packages/environment-capture/financebench/wm_replace_demo.py`): a FIXED agent runs each
held-out task twice — once against the real environment (`adapter.open_env`), once against the
world model behind the same `CommandEnv.execute` seam — and the SAME deterministic
`adapter.grade` scores both. Reported per corpus:

- **Reward agreement rate** — `real_reward == wm_reward` fraction, with the 2×2
  sim-pass/real-pass confusion (per `.agents/docs/proposals/closed-loop-eval-spec.md`).
- **Steps-to-divergence** as a diagnostic (not headline).

The stronger policy-rank agreement question (does the sim rank multiple agents like the real
env — `sim-real-policy-rank-agreement.md`) stays a research direction; the benchmark reserves
the metric name but v1 does not require multiple evaluatees.

### Coverage tiers (honest labeling, v1)

| Corpus | Track A | Track B | License label |
|---|---|---|---|
| bird-sql, dabstep, financebench, crmarena, gaia2, continual-learning | ✓ | ✓ (adapters have `open_env` + `grade`) | financebench/crmarena CC BY-NC (disclosed); gaia2 don't-train notice |
| tau-bench, terminal-tasks, swe-bench | ✓ | later — corpora + fidelity evals exist, but no `CommandEnv` adapter yet (real envs live in `examples/` harnesses) | per existing cards |
| appworld | — excluded (local-only license) | — | — |

## Validation: deterministic core vs the judge (the user's condition)

The deterministic composite is only allowed to be the headline if it **empirically matches the
pinned judge**. Plan:

1. Reuse the meta-eval harness from the judge overhaul (PR #83) and existing prediction sets:
   for every corpus, score the same predictions with (a) the deterministic composite and (b) the
   pinned RubricJudge.
2. Agreement criteria (proposed): per-step score correlation AND — the one that matters —
   **system-level rank concordance ≥ 0.9** (Kendall pairwise concordance across model×condition
   cells, the things the benchmark exists to discriminate).
3. Iterate the extractor specs against real disagreements (AGENTS.md rule 12: read the actual
   outputs, tune on the disagreements).
4. **Fallback**: any corpus whose deterministic composite can't reach the bar keeps the pinned
   RubricJudge as its headline, labeled as such. Track B (reward agreement) is deterministic by
   construction and needs no judge validation.

Judge-pinning discipline throughout (D12): a pinned judge version is part of the benchmark
version; deterministic and judge numbers are never mixed in one column.

## Splits, canary, publishing

- **Splits**: the existing whole-trace seeded splits, frozen test splits SHA-guarded, public on
  HF (`experiential-labs/wmh-<benchmark>-traces`). Benchmark versions pin the split SHAs.
- **Canary (design, implemented in the corpus-publishing follow-up)**: one benchmark-wide canary
  GUID (BIG-bench style) embedded as a metadata field in every published test-split row and
  printed on every dataset card, so any trained model can be probed for test-split contamination;
  the publishing pipeline (envcap `hub_push`) adds it at push time. Rows also carry
  `wmh_benchmark_version`. Republishing datasets with the canary is a follow-up PR on the
  publishing pipeline, coordinated with the corpus owners.
- **Never** fold AgentWorldBench data into any wmh corpus (eval-only, and it would poison our
  own WM rows on their benchmark).

## Reporting

Static results table in `docs/` (version-pinned: benchmark version, split SHAs, judge pin if
any, n per corpus), mirrored on the HF dataset cards. No live leaderboard anywhere (WS-B1
decision). Suggested headline columns per corpus: n · reward-agreement (Track B) · error-flag
accuracy · deterministic composite (or pinned-judge fidelity, labeled, until validated).

## What ships when

- **This PR (design only, per user re-scope 2026-07-07)**: this proposal + the AgentWorldBench
  adapter work (separate deliverable, same PR).
- **Follow-up 1 — meta-eval**: deterministic extractors + agreement study vs pinned judge;
  go/no-go per corpus on deterministic headline.
- **Follow-up 2 — canary**: publishing-pipeline change + dataset republish.
- **Follow-up 3 — Track B harness**: generalize `wm_replace_demo` behind `wmh bench` rows
  (coordinate with the closed-loop eval work, PR #111); tau/terminal/swe adapters as capacity
  allows.
- **Naming/branding**: decided before the first public results table ships.
