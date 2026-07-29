# Latency corner: phase-1 findings

Status 2026-07-27: the canonical tau grid is FILLING (identity arm relaunched 14:05, chunk 0
in flight; truncate and llmlingua2-endpoint staggered behind it). Everything grid-derived
below is marked measured or pending accordingly; the pipeline (`render_latency.py`, latency
metrics in `../common/latency.py`, loaders in `../common/data.py`, statistics per the
binding conventions in `../common/README.md`) renders from whatever chunk or merged matrices
exist and is re-run as arms land. Zero LLM spend; offline computation only.

## Methodology the numbers stand on (fixed before the data arrives)

- **Two clocks, never mixed.** Grid latency is `ScenarioOutcome.call_seconds`: candidate
  MODEL call wall seconds, environment/judge time excluded, summed per task plus compressor
  wall (the scorecard's per-TASK contract, `wmo/optimize/scorecard.py`). Cycle-1 latency is
  whole-episode REAL-tau2 wall clock including the user simulator. Every figure names its
  clock; no number from one appears beside a number from the other without the label.
- **The #295 blank-retry rule.** Per-CALL statistics drop calls the provider answered with
  blank text (`wmo/optimize/report.py::_productive_call_seconds`): blanks return fast and
  deliver no work, so counting them makes a blank-prone model look FAST. Per-TASK statistics
  keep every call: a blank retry's wall time is real time the task took. Kimi-k2.6 blanked on
  24% of calls in the original capture, so this is not a corner case on this pool.
- **Quantiles.** p50/p95 use the scorecard's inclusive method (bounded by observed data). At
  this grid's size (20 test-band scenarios x 2 episodes = 40 tasks per config), p95 is
  roughly "the second-slowest task" — it is reported everywhere but treated as descriptive,
  not as a dominance coordinate.
- **Unscored rows are infrastructure, not verdicts.** They are excluded from quality, cost,
  and latency pools alike, and counted. Partial chunk reads are pre-retry-pass and labeled.
- **Distill-stage verdicts are sourced, never hand-ruled.** A stage annotation quotes the
  recorded gate artifact (cycle-1: `gate.json`, "rejected: after 0.650 regressed below
  before 0.717"); diagnosis prose (the no-teacher-headroom reading) stays in findings, not
  in verdict labels. Once `jt/teacher-gate` lands, the verdict source is
  `wmo optimize distill probe` over the same grid matrices.

## Where tau latency comes from (mechanism now, magnitudes when the grid lands)

Per-task seconds decompose as (number of model calls) x (per-call seconds) + compressor
wall. What is already measured or documented:

- **Episode length is a first-class term, not a constant.** Real tau2 cycle-1 episodes ran
  ~30 messages and p50 82–107 s of episode wall per arm; the grid caps at max_steps=20 with
  per-call model seconds typically single-digit, so a config that takes extra steps to finish
  pays its per-call latency that many more times. Chattiness is a latency lever exactly as it
  is a cost lever (the effective-cost rule's "cheaper-per-token but chattier" warning applies
  to seconds too).
- **Compression trades compressor wall against prefill.** The endpoint compressor bills
  ~+0.67 s per 10k tokens of compressor wall (three-corners DECISIONS entry) against fewer
  provider prefill tokens; net per-task latency is MEASURED per arm, never assumed, and rides
  in `compressor_latency_s` per episode. Until the compaction lane's accuracy verdict lands,
  compaction rungs stay "measured tradeoff, not recommendation".
- **Model choice moves per-call seconds by multiples** (pending grid magnitudes): the pool
  spans serverless open-weight candidates and frontier APIs; the grid's identity arm gives
  the clean per-model read.

## Cold start: kimi-k3's 51-second first call (measured, runner-documented)

K3 is served serverless on Fireworks and its FIRST call after scale-down was measured at
51 s (`run_tau_grid.py` docstring; the runner deliberately has no hang detection, so cold
starts are recorded as wall time, never killed, and the retry pass runs against a warm
model). Consequences for this corner:

- A serverless candidate can have a competitive warm p50 and a catastrophic first-call tail;
  the `cold_start_first_vs_warm` figure separates each episode's first call from the rest so
  the two are never averaged into one flattering number. First-call points are cold-start
  WITNESSES, not clean measurements — idle gaps between chunks are unrecorded.
- For an endpoint MOUNT decision (which is what this corner is), a cold start lands on a
  customer request. A latency-max policy that routes any share of traffic to a scale-to-zero
  backend carries a p99 it never shows at p50; the corner artifact must say so if K3 is in
  its mix.

## The LATENCY-MAX named corner (definition fixed; pick pending data)

Among mountable configs — today a (candidate model x compression arm) pair, i.e. a fixed
policy plus a `CompressionConfig`; routed configs join once the joint-tau master runs the
per-arm fits (never fitted here) — the corner is the config minimizing per-task p50 model
seconds subject to a PAIRED per-scenario reward delta against the best measured config no
worse than the quality floor (`stats.paired_delta`; unpaired mean comparisons are banned by
the common conventions). All three objectives ride on the pick (`latency_max_corner.json`),
along with the paired delta, its CI, and its noise-floor flag, with a sensitivity sweep at
floors of 2 / 5 / 10 reward points below best, because the brief's "acceptable quality" is
not yet a ruled number.

**Question to Silen (also carried in DECISIONS):** what is the acceptable-quality bar for
the latency corner? RECOMMENDATION: 5 points below the best measured config, mirroring the
SLA's "quality within a small tolerance" framing, with the 2/10-point neighbors reported so
the choice is visibly not load-bearing. STEELMAN for an absolute bar instead (e.g. reward
>= 0.7): a relative floor tracks a possibly-mediocre best config, and a customer feels
absolute task success, not distance from our own frontier; cost is that an absolute bar is
arbitrary until the grid says where rewards land.

## Recorded limitation: this corner is an offline mount choice

There is NO online latency-aware routing rule: the knn decision profile has a cost knob and
no latency term, so the latency-max corner selects WHICH policy/candidate mix to mount, not
per-query latency routing. Adding a `lam_latency` term to `knn_decision` is routing-lane
future work with its own 5/5-paired gate, off the canonical result's critical path
(DECISIONS.md 2026-07-27, "THREE-OBJECTIVE CORNERS", gap 3; reaffirmed in the D-DIAL v2
serving-mechanics entry).

## Noise floor and negative results (kept)

- Grid side: 40 tasks per config; a per-config p50 is stable but p95 is ~one episode, and
  reward means carry the WM env's known reward variance. The binding noise floor is ±0.02
  mean reward at these sample sizes (`stats.NOISE_FLOOR_REWARD`, common/README.md): the
  frontier shades the band under the best config, and no finding headlines a delta inside
  it.
- Cycle side: the distill-only line DROPS 71.7% -> 65.0% at cycle 1. That regression is 4
  episodes of 60; the paired sign test over the 7 tasks that moved gives p = 0.45 —
  indistinguishable from noise. The gate's recorded verdict (gate.json) is "rejected: after
  0.650 regressed below before 0.717"; the no-teacher-headroom reading (a 1.6-point
  teacher-student gap left nothing to copy) is the result note's diagnosis, and degeneration
  is ruled out (behavioral metrics flat: 60/60 clean stops, ~30 messages, p50 wall 91 s vs
  82 s). The latency lens adds: warmup distillation did not
  change episode wall time either (93 s vs 94 s mean) — no latency win came from cycle 1.
- Per-episode COST was not recorded in cycle-1 artifacts (run total $34.94), so the stage
  chart's cost annotation is honest about its absence rather than derived after the fact.

## Figures and artifacts in this directory

| artifact | deliverable | state |
| --- | --- | --- |
| `training-stage.{png,json}` | shared stage-vs-quality chart, p50-annotated (canonical `common/ablation_chart.py`, latency lens) | rendered (cycle-1 real data; WM panel + ablation lines named pending on-figure) |
| `latency_per_config.{png,svg}` | per-config per-task p50/p95 | pending grid chunks |
| `latency_quality_frontier.{png,svg}` | latency-quality frontier, cost as marker area | pending grid chunks |
| `cold_start_first_vs_warm.{png,svg}` | K3 cold-start callout | pending grid chunks |
| `latency_max_corner.json` + `config_points.json` | the named corner + auditable table | pending grid chunks |

Re-render: `uv run --extra viz python .agents/docs/research/corners/latency/render_latency.py`
(`--synthetic` smokes the whole pipeline against a fake matrix, writing under `figures/synthetic/`).
