# COST-MAX corner: where the savings actually come from

> Status 2026-07-28: ALL THREE matrices FINAL (440/440 scored each). All three per-arm
> routed rungs measured (fits + dial replays on the shared held-out 6). Cost deltas carry
> paired cluster-bootstrap CIs (see the significance audit at the end). Remaining: the
> real-episode leg + WM-vs-real probe (master). Publication gated (DECISIONS 2026-07-27).

## The accounting rule every number obeys

All cost aggregation on this page comes from `wmo.optimize.scorecard` and nothing else. It
implements the binding D-COMPRESS rule: **cache-adjusted effective cost per COMPLETED
task**, compressor/router inference folded in as overhead, unscored episodes excluded from
numerator and denominator and their spend reported, money conserved on the artifact.
Per-token price is not cost: an arm that fails half its tasks is cheap per run and ruinous
per completed task, and the compaction lane already measured the inversion this metric
exists to catch (below).

Every number is labeled `measured` or `estimate`, `wm_simulated` or `real_episode`, with
the judge named. The grid's WM cells are judged by rubric-v2 pinned on opus-4-8; cycle 1's
training numbers are real tau2 episodes under tau2's own reward.

## The savings decomposition (the honest shares)

Savings behind an optimized endpoint can come from four places. Their current shares:

1. **Model-selection share: today this is the WHOLE measured savings story** [measured,
   wm_simulated, judge rubric-v2 (opus-4-8), 20 scenarios x 2 episodes]. Against the named
   fable-5 anchor ($0.958 per completed task here), the single-model swap already clears
   the SLA band: opus-5 is **-59.0% effective cost at +10.9 pt quality** (paired CI +0.3 to
   +21.9 pt, excludes zero: better AND cheaper), and gpt-5.5 is **-59.2% at +2.8 pt**
   (CI -7.6 to +12.6, parity). Cheaper models buy more savings only by paying quality:
   glm-5.2 -80.5% at -5.8 pt, haiku-4-5 -89.8% at -14.3 pt, gpt-5.4-mini -93.0% at
   -34.1 pt. The routed rung on top of this is measured (see the routed bullet in the 40%
   section): the router discovers opus-5 automatically and holds best-single parity; the
   dial's savings leg does not pay on tau's ~14-scenario bank, per the small-bank law. On
   routerbench-ours9 (different corpus, quoted as measured there, never blended), the
   dial runs -13.9% (quality max, +1.14 pt) to -46.2% (max savings, -0.54 pt) vs ITS best
   single model.
2. **Compression share [measured, FINAL matrices; SIGNIFICANCE AUDIT 2026-07-28 applied]:
   compression on tau buys no cost win and SIGNIFICANTLY hurts quality on the anchor
   model.** With paired cluster-bootstrap CIs on the cost deltas (n=20): truncation's
   fable-5 inversion is **+47.1%, CI -21.9 to +175.1** - the point estimate is a large
   cost increase but the tau data alone does NOT resolve it (per-scenario cost variance is
   huge); the independent financebench result (+21-36% on matched controls) remains the
   resolved evidence for the inversion, and tau is directionally consistent. The learned
   llmlingua2-endpoint's fable-5 cost delta is **+14.1%, CI -29.6 to +147.3** (unresolved),
   while its QUALITY harm IS resolved: **-8.3 pt, CI -16.4 to -1.4**. So the defensible tau
   statement is: compression shows no evidence of saving money on strong models and
   measurably costs quality on the anchor; the learned compressor's cost damage is far
   below the dumb control's in point estimate. (Earlier +146.2%/+23.6% figures were
   pre-repair artifacts, retracted; and this section's earlier "CONFIRMED on tau" phrasing
   overclaimed - financebench confirms, tau is consistent-but-unresolved on cost.) The financebench mechanism (deleting load-bearing observations fails tasks and
   lengthens episodes) still shows in truncate's latency column (p50 +170-480% on several
   models). Per-token accounting would have called every arm cheaper; effective cost per
   completed task catches the inversion. Compaction rungs stay "measured tradeoff, not
   recommendation" pending that lane's accuracy verdict.
3. **Student share (distillation): zero, and zero BY PROGRAM DECISION.** Silen's ruling
   (2026-07-28, this chat): distillation is not being pursued. Cycle 1's gate had already
   REJECTED the warmup adapter (teacher Qwen3.6-27B 73.3% vs student base 71.7% at k=3: no
   headroom; before-vs-after p=0.45 at n=60), and no unpromoted student ever enters the
   pool, so no served token is cheaper because of training and none is planned to be. The
   distill-only rung reads "no measurable effect at this sample size", never a lift and
   never a regression. The teacher-gate verdict on this grid (DISTILL: cheapest sufficient
   teacher gpt-5.5 at $0.3907/completed task, keeping 82% of opus-5's +45 pt gain over
   gpt-5.4-mini) is recorded in `numbers.json` as the repo function's descriptive output,
   NOT as a plan. Teacher-selection economics are OWNED
   by `wmo.optimize.teacher.select_teacher` (#329, merged): this page cites its verdict and
   never hand-computes teacher economics. PRICE-ORDERING REVIEW against this corner's
   scorecard conventions (2026-07-27, cost chat): ALIGNED. The primary ladder IS
   `scorecard.effective_cost_per_completed_task` over each model's own rows, so it inherits
   cache-adjusted pricing, the per-completed-task denominator (a chatty teacher that fails
   tasks ranks expensive), unscored-row exclusion, and measured-$0-is-missing-not-free; the
   fallback is a LIST-price ordering key (input+output per Mtok), used only when any model
   lacks a measured figure, applied to the WHOLE ladder (one basis, never mixed) and stamped
   on the verdict as `price_basis`. One caveat worth carrying: the list-price fallback is
   not cache-adjusted, so on cache-dominated workloads two adjacent models could in
   principle order differently than their real serving cost; any verdict quoted here with
   `price_basis="list"` says so. The z=1.96-at-n=8 nit is the master's logged note; it
   matches the program-wide convention. Final-grid verdict (cited verbatim in
   numbers.json): DISTILL, cheapest sufficient teacher gpt-5.5; recorded as descriptive
   output only per the no-distillation ruling. (The partial-grid read earlier that day was
   INSUFFICIENT EVIDENCE at 5 shared scenarios: the gate tightening as evidence arrived is
   the mechanism working.)
4. **Cheap-model-was-already-fine share (the anchor's weakness)**: the share of "savings"
   that any cheap model would have delivered because the anchor is overkill for part of the
   workload. This is real money but not an optimizer achievement, which is why every delta
   is reported against BOTH anchors (next section).

## The two-anchor discipline (inherited from the routing lane)

RULING (Silen, 2026-07-28, this chat): **fable-5 is the anchor for ALL baselines, routed
configs included**; the dominated-anchor situation is understood and accepted. The
best-single column stays on every artifact as the check, per the rest of this section.

### The "is anything cheaper than opus-5" check (Silen ask, measured)

Seven models are cheaper than opus-5 per completed task on this grid; every one pays
quality for it. Five are RESOLVED worse (paired CI excludes zero): gpt-5.4-mini (-82.8%,
-45.0 pt), haiku-4-5 (-75.0%, -25.2 pt), deepseek-v4-pro (-67.5%, -29.0 pt), glm-5.2
(-52.4%, -16.6 pt), opus-4-8 (-8.6%, -15.0 pt). Two are UNRESOLVED at this sample size
(CI spans zero): **kimi-k2.6** (-10.0%, mean -11.8 pt, CI -28.5..+4.5) and **gpt-5.5**
(-0.5%, mean -8.1 pt, CI -18.1..+1.5). Neither is a measurable cheaper-at-parity win:
gpt-5.5's saving is negligible, and kimi-k2.6's real 10% saving comes with p50 246s per
task vs opus-5's 38s (6.5x slower) plus an unresolved but negative-leaning quality mean.
Resolving kimi-k2.6 properly needs more episodes than the cohort pin allows; recorded as
an open cell, not bought unprompted.

Savings quoted against a weak anchor overstate. The headline anchor is **fable-5** (named,
$5/$25 per Mtok class): the frontier reference a customer would otherwise be paying for.
Alongside it, every config is also scored against the **best single pool model by mean
reward on this grid** ([PENDING: name]; on ours9 the equivalent baseline was the bar the
dial's -46.2% was measured against). If the best single model is much cheaper than fable-5
at similar quality, the fable-5 column reads large for reasons that are partly
model-market facts, not optimization; the best-single column is the defensible optimizer
contribution. Both columns ship in `numbers.json` and on the charts.

## What the 40+% target reads as against measurement

The SLA promise is "always at least ~40-50% cheaper than the frontier reference at quality
within a small tolerance".

- Against the **fable-5 anchor** on tau [measured]: CLEARED with headroom by single-model
  selection alone. gpt-5.5 delivers -59.2% at quality parity (+2.8 pt, CI spans zero);
  opus-5 delivers -59.0% while being BETTER (+10.9 pt, CI excludes zero). The 40-50% band
  is not the frontier of this workload; it is comfortably inside it.
- Against the **best-single anchor (opus-5)** [measured], the honest optimizer-contribution
  reading: NOTHING measured beats it yet. opus-5 dominates this grid (it is the quality
  max AND cheaper than fable-5 at $0.393/task), so every other single model saves money
  only by losing quality (gpt-5.5: -0.5% at -8.1 pt), and the compressed arms lose on both
  axes for strong models.
- **The routed rung [measured, identity-arm fit, 6 held-out scenarios, embedding replay
  ~$0.001 logged per the 2026-07-28 spend ruling]**: at the quality and balanced dials the
  router routes ALL six eval scenarios to opus-5, i.e. it automatically discovers the
  dominant model and delivers best-single parity (-67.3% vs fable-5 at +12.5 pt on ITS
  eval band; CI spans zero at n=6; NOT comparable to the 20-scenario single-model numbers,
  different scenario set). Past the balanced point the cost leg routes one scenario to
  kimi-k2.6 and gets WORSE on both axes vs opus-5 (+13.2% cost, -6.25 pt): on a
  ~14-scenario bank the dial's savings leg does not pay, exactly what the small-bank
  caveat predicted. The routing rung's honest tau claim is **cost-at-parity with the best
  single model via automatic discovery**, not savings beyond it (ours9's 1199-scenario
  bank is where the savings leg earns -24.7% to -46.2%).
- HONESTY NOTE the headline must carry: fable-5 is a WEAK anchor on this workload; it is
  dominated by opus-5 on both axes. A "-59% cheaper than frontier" claim that names
  fable-5 is true and clears the SLA, but the defensible optimizer contribution is the
  best-single column, where the measured answer today is "pick opus-5, which the router
  finds automatically". RULING: fable-5 leads all baselines (Silen 2026-07-28); this
  column stays adjacent as the check.
- The sales frame stays "run 10x more for the same budget" (an **estimate** derived from
  measured per-task cost, and labeled as such wherever quoted; vs fable-5 the measured
  multiplier at parity is ~2.5x, not 10x, on this workload).

## Standing caveats on every number here

- WM-simulated cells are ~85% telecom (the corpus mix); the real-episode leg (pinned
  balanced 20 through the served endpoint) is the check, and the master's WM-vs-real probe
  decides how much this corner's analysis the WM can carry alone.
- Latency figures are per-task MODEL seconds (call_seconds excludes env/tool time), so they
  understate wall clock and flatter prompt-shortening optimizers; they are read jointly
  with cost per completed task, never alone.
- 7 of the 20 holdout tasks include tau2's NL-assertion judge in their reward basis.
- Episodes=2 per cell: per-cell variance is halved vs e1 but the noise floor on per-model
  quality deltas remains material at n=20 scenarios; paired stats over shared cells, not
  headline means, carry the load-bearing claims.

## Artifacts

- `lens.py`: this corner's declarative figure spec, rendered by the ONE shared runner
  (`common/build_corners.py`, charter Amendment): scorecard-only cost aggregation,
  paired-delta evidence on every quality claim, the distillation verdict cited verbatim
  from `wmo.optimize.teacher.select_teacher`. Rerun
  `uv run python .agents/docs/research/corners/common/build_corners.py --lens cost` as
  grid-c2 cells land (live #330 sidecars load before any matrix merges); routed rungs
  attach via `rows_for_policy` when the master's per-arm fits are delivered.
- `figures/dial_cost_curve.png`: ours9 dial anchors as measured (tau panel pending fits).
- `figures/training_stage_cost_lens.png`: the shared training-stage chart, cost lens
  (cycle 1 as measured, real_episode; cost deltas attach when the grid's student cells
  land).
- `figures/savings_vs_fable5.png`, `figures/effective_cost_per_task.png`: [PENDING grid].
- `numbers.json`: every computed figure with provenance and the scorecard's own
  cost-assumptions sentence per entry.

## Significance audit (2026-07-28, paired cluster-bootstrap CIs on BOTH axes)

What survives a CI-excludes-zero bar today, on the FINAL grid:

- **RESOLVED, the headline**: routed/opus-5 vs fable-5 cost (-59.0%, CI -74.1..-39.4 at
  n=20; routed replay -67.3%, CI -84.9..-44.6 at n=6) AND opus-5's quality gain (+10.9 pt,
  CI +0.2..+21.9). gpt-5.5's -59.2% cost is resolved (CI -72.5..-42.6) with quality parity
  (CI spans zero, which IS the parity claim). llmlingua2's quality harm on fable-5 is
  resolved (-8.3 pt, CI -16.4..-1.4).
- **LOO-CV UPGRADE (the power fix, run 2026-07-28)**: leave-one-out over all 20 scenarios
  (20 fold-fits, every scenario routed by a fit that never saw it, same recipe via
  fit_knn_policy's own scaling; ~cents embeddings). The ROUTED rung at n=20: **cost vs
  fable-5 -57.1%, CI -72.7..-37.3 - RESOLVED**; quality +9.0 pt, CI -0.9..+19.8 (still
  leaning-better-not-resolved: the claim stays at-least-parity). vs opus-5 the router is
  within +4.7% cost (CI +0.0..+17.1) and -1.9 pt (CI -5.6..+0.0): cost-at-parity with
  best-single, now at n=20. DISCOVERY REPLICATES: 20 of 20 independent fold-fits chose
  opus-5 as fallback and routed 90-95% of held-out scenarios to it at every detent. The
  single-split kimi-k2.6 delegation harm (+13.2%/-6.25 pt) was substantially a
  one-scenario artifact: pooled across folds it shrinks to an unresolved +4.0%/-1.75 pt
  drag. Dial position barely moves the LOO mix (19-18/20 opus-5 at every detent): on a
  ~19-scenario bank the savings leg is inert, the small-bank law again.
- **UNRESOLVED at this n**: every compression COST delta on strong models (see above); the
  routed rung's +12.5 pt quality edge over fable-5 (n=6, CI -4.2..+29.6; its claim stays
  cost-at-parity-or-better); the kimi-k2.6 delegation's harm vs opus-5 (borderline: cost
  CI +0.0..+57.2, quality CI -18.8..+0.0, both touching zero at n=6); everything about
  routed-compressed-kimi-k3 beyond "no evidence it beats opus-5" (CI -41.7..+502.8).

The power plan, cheap to expensive: (1) DONE - cost CIs on every delta (offline, free).
(2) Leave-one-out CV of the fits: refit on 19, replay the held-out 1, x20 - turns the
routed rung's n=6 into n=20 with zero episode spend (~cents of embeddings). (3) An
episode-addendum cohort on the decision-relevant cells (test band x {opus-5, kimi-k2.6,
fable-5 compressed} x +4 episodes ~ 240 cells ~ $120-160) to resolve the compression cost
inversion and the delegation question on tau itself - needs a Silen cap + master cohort
discipline. (4) The real-episode leg (planned, master's task) is the independent check,
not a power fix (same n=20 pin).

## The Pareto answer (Silen ask, 2026-07-28; n=20 basis, all 11 pool models)

Single-model frontier: gpt-5.4-mini ($0.067, -34.1) -> haiku-4-5 ($0.098, -14.3) ->
glm-5.2 ($0.187, -5.8) -> kimi-k2.6 ($0.353, -0.9) -> gpt-5.5 ($0.391, +2.8) -> opus-5
($0.393, +10.9). fable-5 ($0.958, the anchor) is deep inside the frontier. The ROUTED
endpoint ($0.411, +9.0, LOO n=20) sits a hair inside the frontier: weakly dominated by
opus-5 on point estimates, with the gap (+4.7% cost, -1.9 pt) statistically
indistinguishable from zero. That gap is the discovery tax, and it is the honest price of
not knowing ex ante that opus-5 is the corner. On tau the dial does not trace the
frontier (the LOO mix pins to opus-5 at every detent; small-bank law); the frontier's
cheap end is reachable only as static mounts of the cheap models. RULING logged: compaction
analysis skipped going forward (rungs stay as recorded evidence); routing is the focus.

## The real-episode leg (2026-07-28): THE QUALITY HEADLINE BREAKS IN REALITY - probe STOPPED per the pre-registered rule

Setup [real_episode, tau2's own reward, canonical pins, fresh capture cohort, pinned 20 x 2
episodes per arm]: the identity policy served at balanced (routing verified live:
x-wmo-routed-model opus-5, D-DIAL config correct) - but the endpoint 501s tool calls on
anthropic-kind entries (product finding, serving lane), so the routed arm executed ITS
DECISION direct (opus-5; the balanced policy routes 100% there, verified live + 20/20 LOO
folds). Deviation labeled on every row. Telecom required the full-task-split runner fix
(committed 710e6582).

THE VERDICT (paired per scenario, 40/40 episodes scored per arm):

- QUALITY BREAKS, with the SIGN INVERTED: real routed-vs-fable-5 is **-25.0 pt
  (CI -42.5..-5.0, sign test p=0.012, 1 up / 10 down / 9 tied)** where the WM said
  **+10.9 pt (CI +0.3..+21.9)**. On real tau2, fable-5 is the quality king (0.775 mean
  reward, 31/40 success) and opus-5 collapses (0.525, 21/40). Both legs are resolved;
  they disagree in SIGN at the top of the pool.
- COST DIRECTION HOLDS, magnitude shrinks: routed real $1.092/completed vs fable-5
  $1.456 = **-25.0%** (the WM's -57.1% overstates real savings ~2.3x, partly because the
  quality collapse shrinks the completion denominator).
- Sign agreement is uninformative at this n (3/5 after 10 ties; 15/20 scenarios bridge
  via row provenance, 5 do not map into the WM matrix ids - carried as a caveat).
- DECISION RULE FIRED: quality at-least-parity broke, so the probe STOPPED (no phase-2
  compressed config, no expansion). Bill: real episodes $68.06 candidate-side (fable-5
  $45.13, routed $22.93) + qwen addendum $30.70.

WHAT THIS MEANS FOR EVERY NUMBER ABOVE: the wm_simulated headline ("routed -57% at
at-least-parity vs fable-5") is a SIMULATION claim and must not ship as a real one. The
real, measured tau statement today: routing-to-opus-5 saves 25% per completed task and
loses 25 quality points vs fable-5. The WM top-of-pool ranking inverts against reality;
candidate mechanisms (not adjudicated here): rubric-v2-on-opus-4-8 judge affinity for the
same-family opus-5, the WM's 20-step cap vs real 100-turn episodes (opus-5's WM episodes
were conspicuously short at ~11 steps), and WM environment fidelity on long-horizon tool
use. Corroboration that the WM misranks against real on a second model: qwen3.5-9b scores
0.46 mid-pack in the WM yet solved 71.7% on real tau2 in cycle-1's cohort (vs fable-5's
0.775 real here; different cohorts, noted not pooled).

RECOMMENDED NEXT MEASUREMENTS (masters/Silen decide; probe spend stops here per the rule):
(1) real qwen3.5-9b on this probe cohort (~$3-6 at its prices; if near-parity holds in
THIS cohort, the real Pareto winner is a $0.10/$0.15 model and the product story flips to
"we found a 50x cheaper parity model"); (2) a judge-affinity check (rescore a sample of WM
episodes with a non-Anthropic judge); (3) refit the policy on real-validated evidence for
this workload.

### The discrepancy, diagnosed (2026-07-28, zero-spend forensics)

The WM judge cannot recognize correct inaction. On the 8 bridgeable scenarios where
fable-5 scored a perfect 1.00 on real tau2 (the explain-policy-and-do-nothing task class:
basic-economy cancellations, post-booking insurance), the WM judge scored the same model
0.00-0.57 with critiques of the form "you only retrieved the reservation details and took
no action" - on transcripts showing correct policy refusals. The judge zeroes correct
refusal for every model (opus-5's WM rows on the same tasks: 0.00 vs real 1.00), so the
inversion is not family favoritism: clipping the refusal class deletes fable-5's real
advantage from the sim. Ruled out by the same pass: truncation (episodes end normally in
both worlds), scaffold early-quit (uniform), judge parse fail-close (real but rare, 0-2
rows/model). One open verification: config.toml names haiku-4-5 as judge_model while the
program has quoted opus-4-8 - the master confirms the actual grid judge. The router is
exonerated: it optimized correctly against wrong rewards. Fixes filed in DECISIONS
(inaction clause / end-state verification, real-evidence refit, /improve-judge with the 8
divergent scenarios as the labeled regression set).
