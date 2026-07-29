# The canonical tau-bench result: routing, compaction, and distillation, jointly

> DRAFT - numbers marked [PENDING] slot in from the grid, the corner analyses, and the
> real-episode validation. Publication is gated (DECISIONS 2026-07-27) on the three corner
> analyses and the cost corner's verdict. Nothing in this document may be quoted externally
> until that gate lifts and this banner is removed.

## The claim this document substantiates

world-model-optimizer takes an agent's traces and returns an OpenAI-compatible endpoint that
serves the same workload cheaper at comparable quality. Behind the endpoint, three optimizers
compose: a learned inference policy (routing) chooses a model per conversation, compaction
compresses request context before the call, and distillation trains a small personal model
that earns traffic as it improves. This document is the measured, end-to-end demonstration of
that composition on one public benchmark, with every number's provenance labeled and every
negative result kept.

## Why this is credible (the method, before the numbers)

1. Simulated and real are never blurred. The world model built from tau-bench traces supplies
   the closed-loop evidence for FITTING policies; every headline quality/cost claim is then
   checked on REAL tau2 episodes through the same serving path a customer uses, under a
   pre-registered pass/fail rule. Numbers below carry [wm] or [real] tags. This check has
   TEETH: the earlier sim-to-real result (the sim picks the real winner, errs pessimistic)
   did NOT survive at the top of this pool - the real-episode section below reports the WM's
   quality ranking inverting against reality, and the claims re-scoped accordingly.
2. Cost means cache-adjusted effective cost per completed task. Not per-token price: a model
   that is cheap per token and fails tasks, or one that forfeits the provider prompt cache, is
   expensive in the only sense that matters. The metric folds the compressor's own inference
   bill and reports unscored spend rather than hiding it. (The dumb-compression controls in
   the companion accuracy study RAISED effective cost while lowering per-token cost - the
   metric exists because that inversion is real.)
3. The controls are matched and the gates are pre-registered. Compression arms include a
   truncation control matched on ACHIEVED token ratio; the distillation gate and its holdout
   were fixed before any training spend; the router's guard makes its worst case the fallback
   model by construction.
4. Failure is reported as failure. The first distillation cycle was REJECTED by its own gate -
   the teacher had no headroom over the student on this benchmark - and that verdict, its
   $34.94 cost, and its diagnosis are part of this record, not an embarrassment excised
   from it.

## The pipeline (one command per stage; the cookbook walks it)

traces.otel.jsonl -> `wmo build` (the world model) -> `wmo providers set` (the candidate pool)
-> `wmo optimize model <name> [--compressor <id>] [--distill <config>]` (measure, fit, tune,
report) -> `wmo serve` (the endpoint). See docs/cookbook/tau-bench.md for the artifact-by-
artifact walk. Every stage in this study ran through those public commands or the library
functions they wrap.

## The evidence

### The headline table (cross-benchmark; every cell from the corners' numbers.json)

| Benchmark | Routed config | Quality | Effective cost/task | Basis |
|---|---|---|---|---|
| tau-bench (multi-turn tool agent) | 100% opus-5, discovered by 20/20 LOO fold-fits; guard refuses to degrade | +9.0 pt vs fable-5 (CI -0.9..+19.8: at-least-parity, leaning better) [wm] | **-57.1%** ($0.411 vs $0.958; CI -72.7..-37.3, RESOLVED) [wm] | LOO-CV n=20, judge rubric-v2 (opus-4-8) |
| terminal-tasks (one-shot bash agent) | 100% sonnet-5 (constant policy; kimi-k3 ties 0.949 vs 0.948, broken on price) | -0.4 pt vs fable-5, within noise (paired CI touches 0) [wm] | **-69.4%** ($0.011 vs $0.035; CI -77..-61) [wm] | held-out 6, full pipeline via public commands |
| routerbench-ours9 (1,199 QA prompts, 9 models) | frontier-pinned mix (37% gpt-5.5 / 33% sonnet-5 / 23% fable-5) | +0.84 pt over best single (0.978 vs 0.969) | -35.0% ($0.0023 vs $0.0035) | 360 held-out; measured on ours9, never blended with tau |

**Read the tau row with the real-episode section below**: its quality cell is [wm] and the
real probe INVERTED it (-25.0 pt real vs the WM's +9.0..+10.9); its cost direction held at
-25% real (not -57%). The terminal and ours9 rows have no real leg yet.

The three-stage figures behind this table (model-selection stage, compression stage,
distillation stage per benchmark): `corners/cost/figures/three_stage_tau.png`,
`corners/tb2-cost/figures/three_stage_terminal.png`, `corners/cost/figures/three_stage_ours9.png`,
all rendered by the ONE shared runner from the final matrices.

### The ablation ladder (measured; the grid behind the tau row)

Rungs vs the fable-5 anchor on the same 20 held-out scenarios x 11 candidates x 2 episodes
per arm [wm], judge rubric-v2 (opus-4-8), cohort f1ebaca6, 1320/1320 cells scored:

- **Model selection carries the measured savings.** opus-5 single: -59.0% cost (CI
  -74.1..-39.4) at +10.9 pt (CI +0.2..+21.9, resolved better AND cheaper). gpt-5.5:
  -59.2% at parity. fable-5 sits deep inside the single-model Pareto frontier.
- **+routing = automatic best-single discovery plus a guard.** The LOO-CV routed rung
  (-57.1%, resolved) is within +4.7% cost of knowing opus-5 ex ante (CI +0.0..+17.1) -
  the discovery tax. On a ~19-scenario bank the dial's savings leg is inert (the
  small-bank law): tau's routing claim is cost-at-parity via discovery, not accuracy lift.
- **+compaction on tau: no cost win, resolved quality harm on the anchor.** Truncation
  control +47.1% cost (CI -21.9..+175.1, unresolved on tau alone); learned
  llmlingua2-endpoint +14.1% (CI -29.6..+147.3, unresolved) with quality harm resolved
  (-8.3 pt, CI -16.4..-1.4). The resolved evidence for the cost inversion is the
  companion financebench study (+21-36% on matched controls); tau is directionally
  consistent. Under compressed serving the discovered best-single FLIPS (kimi-k3 replaces
  opus-5 in the llmlingua2 arm's fit) - compaction changes the frontier, not just the
  bill. Compaction rungs stay "measured tradeoff, not recommendation".
- **Repair provenance**: the compressed arms' first pass lost 26%/57% of cells to a
  compressor-endpoint outage and local connection exhaustion; all 368 cells were re-bought
  under the same cohort pins (the earlier +146.2%/+23.6% inversion reads were pre-repair
  artifacts and are retracted in the corner's findings).

### Quality across training stages [PENDING: the shared corners chart]

Student quality by training stage with the three lever ablations. Cycle 1 (teacher
Qwen3.6-27B) appears as measured: no promotion - teacher 73.3% / student-before 71.7% /
student-after 65.0% at k=3 [real], paired sign test p=0.45, verdict "no measurable effect at
this sample size; nothing to distill from a peer teacher". There are no later stages:
distillation is not pursued (program ruling, 2026-07-28), and the teacher-search gate is now
repo code (`wmo optimize distill probe`, PR #329) whose verdicts on these grids are recorded
as descriptive output only. Chart: `corners/cost/figures/training_stage_cost_lens.png`.

### The corners program (amended)

The quality-max and latency-max analyses are suspended (their lens specs and findings are
frozen on the branch, resumable); the COST corner drives, on two benchmarks (tau +
terminal-tasks), through one shared runner (`corners/common/build_corners.py`) so a number
appearing twice is the same computation. Every chart still reports all three objectives.
The latency limitation stands: no online latency-aware routing rule exists yet; latency
corners are offline mount choices.

### Real-episode validation: THE QUALITY HEADLINE BREAKS, AND THIS DOCUMENT SAYS SO

The probe ran (pinned 20 scenarios x 2 episodes per arm, canonical pins, tau2's own reward,
40/40 scored per arm) and its pre-registered rule FIRED [real]:

- **Quality inverts in sign at the top of the pool.** Real routed-vs-fable-5 is **-25.0 pt
  (CI -42.5..-5.0, sign test p=0.012)** where the WM said +10.9 (CI excludes zero). On real
  tau2, fable-5 is the quality king (0.775) and opus-5 collapses (0.525).
- **The cost direction holds, the magnitude does not**: -25.0% real per completed task
  where the WM said -57.1% (a ~2.3x overstatement, partly because the quality collapse
  shrinks the completion denominator).
- Consequence, applied throughout this document: every wm_simulated tau number is a
  SIMULATION claim and is never quoted as a real one. The measured real tau statement
  today: routing-to-opus-5 saves 25% per completed task at a 25-point quality loss vs
  fable-5. Candidate mechanisms under investigation (not adjudicated): judge affinity
  (rubric-v2 on opus-4-8 scoring same-family opus-5), the WM's 20-step cap vs real
  100-turn episodes, WM fidelity on long-horizon tool use.
- The live lead: qwen3.5-9b scored mid-pack in the WM yet solved 71.7% of real tau2 in
  cycle-1's cohort - if near-parity replicates on this probe cohort (~$3-6 check,
  authorized), the real Pareto winner is a ~50x-cheaper model and the product story
  becomes cheap-parity discovery. [PENDING: qwen real probe + judge-affinity check.]
- Deviation labeled on every row: the endpoint 501s tool calls on anthropic-kind entries
  (serving-lane finding), so the routed arm executed its verified decision (100% opus-5)
  direct.

This section is the reason the method section says what it says: the probe rule was fixed
before the run, it fired, spending stopped, and the inversion is reported at the same
prominence as the savings. A reader who trusts anything here should trust it BECAUSE this
section exists.

### The compound loop: reported as designed, not as achieved

The plot this product would sell - the router measurably shifting traffic to an improving
personal model - has NO measured point, and this document says so plainly. The
infrastructure exists end to end (train -> gate -> pool entry -> refit -> serve), and the
gate is repo code that runs on data the router already buys (`wmo optimize distill probe`).
On tau it measured no teacher headroom and refused (cycle 1, $34.94, kept); distillation
was subsequently not pursued program-wide. The gate refusing to promote a non-improvement
IS the mechanism working; a measured traffic-share shift requires a workload with real
teacher headroom, deliberately left for the future.

## Honest limitations (standing, whatever the numbers say)

- tau-bench's corpus yields 20 distinct held-out scenarios; the routing bank is small, so the
  routing rung's claim is cost-at-parity under a guarded policy, not accuracy lift. The
  evidence-volume law from the routing research (routability emerges near ~1000 scenarios)
  says larger corpora are where routing accuracy gains live.
- The WM-simulated ladder is ~85% telecom (the corpus's mix); the real-episode leg is the
  balanced check.
- 7 of the 20 holdout tasks include tau2's own NL-assertion judge in their reward; rows
  record the fully-deterministic subset separately.
- The dial's five measured anchors were calibrated on routerbench-ours9 and are quoted as
  such until re-measured jointly on this grid (D-DIAL v2).
- Sim-to-real agreement is quoted as per-scenario paired sign agreement; model-mean rank
  correlations at n<=9 models sit inside their null noise band and are descriptive only.

## Reproduce (the pins)

- Grid cohort: `.wmo/jt/grid-c2` at repo tip f1ebaca6, runner `.agents/scripts/run_tau_grid.py`
  (+ the repair seeder, #348), pool = the cohort's pinned `pool.toml` copy (11 candidates),
  20 test-band scenarios x 11 models x 2 episodes x 3 arms = 1320/1320 scored, max_steps 20,
  history_chars 2000, judge rubric-v2 pinned on opus-4-8. All-in cost $943.86.
- Compression arms ratio-matched by the persisted calibration (llmlingua2-endpoint v1 @ 0.5
  keeps 0.5656; truncate @ 0.33 keeps 0.5604).
- Fits: `wmo optimize route fit`, knn, z=0.5, rag_num=7, min_pairs=3, rag_thres=0.95,
  floor_q=0.05, se_floor on, embedder azure text-embedding-3-large 3072d; identical 14/6
  fit/held-out split across arms (verified); compressed arms fitted with their exact served
  compression stamped (`fit_compression`); LOO-CV = 20 fold-fits, each scenario routed by a
  fit that never saw it. Artifacts: `.wmo/jt/grid-c2/<arm>/policy.json` + `.bank.npz`.
- Real leg (in flight): canonical tau2 pins - max_turns 100, timeout 1800, max_tokens 8192,
  user sim gpt-5.4-mini, retries 0, fresh capture-cohort label.
- Every aggregated number: `wmo.optimize.scorecard` only, via
  `corners/common/build_corners.py`; per-record provenance in each corner's `numbers.json`.

## Provenance of this document

Written by the joint-tau integration effort, 2026-07-27. Companion documents:
docs/cookbook/tau-bench.md (the how-to), docs/usage.md (the CLI map),
docs/research/world_model_findings.md (the layer-by-layer research record).
