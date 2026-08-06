---
area: Form-factor launch (FF4)
status: Proposal — v1 domains chosen, capture architecture pending sign-off
created: 2026-07-05
---

# FF4 — the world model as a pre-run estimator

**Thesis.** Any process whose real measurement loop is slow, expensive, or dangerous can be
compressed by a world model trained on traces of real runs: propose a configuration
(action = structured tool call), get a predicted measured outcome (observation = numbers +
failure states) *before* running it. A "trace" is a sweep session, so retrieval conditions on
sweep history — the WM does grounded in-context regression **with provenance**: every estimate
traces back to real runs on *your* hardware, not a textbook formula. This is the entry point to
the manufacturing/hardware wedge: physical testing cycles are slow and expensive; companies need
cheaper counterfactual testing before building.

## The domain taxonomy

Every entry has the same shape: **config → measured outcome (+ failure cliff, if any)**, with a
ground-truth tag: `[have-now]` measurable on hardware we own (M3 Max 128 GB / MPS, 2×H100 dev
box), `[public-data]` distillable from a published dataset, `[synthetic]` generatable from a
real toolchain, `[acquire]` needs partner data or lab access.

### 1. Performance & resources

- **ML training memory / OOM / throughput** `[have-now]` — model dims, batch, seq len, dtype,
  optimizer, grad-checkpointing, device → peak memory, tokens/sec, **OOM-or-not**.
  *Useful:* right-size the GPU before renting it; predict the OOM before launching the
  multi-hour job; the failure cliff is where analytic formulas systematically miss (allocator
  overhead, fragmentation, MPS/CUDA context).
- **LLM inference serving** (vLLM) `[have-now, H100]` — batch, context length, quantization,
  KV-cache config, tensor parallel → throughput, TTFT/ITL latency, **CUDA OOM**.
  *Useful:* capacity planning and "will this model fit at this context length" answered without
  burning GPU-hours; the most quotable form of the launch ("predict the OOM before you rent the
  GPU").
- **FPGA synthesis + place-and-route** (yosys/nextpnr) `[have-now, synthetic-from-real-tool]` —
  HDL design parameters (width, depth, pipeline stages, target part) → LUT/BRAM/DSP usage,
  **fits-on-chip-or-not**, max clock MHz. *Useful:* P&R is the minutes-to-hours loop hardware
  engineers actually wait on; predicting "fits + closes timing" before the run compresses EDA
  iteration. The most genuinely hardware-flavored corpus we can ground-truth today, and the
  gateway to full chip-design PPA estimation (OpenROAD/OpenLane).
- **Distributed training config** `[have-now, 2×H100]` — DDP/FSDP/TP sharding, gradient
  accumulation → step time, comms overhead, memory per rank. *Useful:* pick the cluster layout
  before the multi-node rental.
- **Embedded firmware size** (arm-none-eabi-gcc for MCU targets) `[have-now]` — features,
  optimization flags, target MCU → .text/.data/.bss bytes, **fits-in-flash-or-not**.
  *Useful:* flash/RAM exhaustion is embedded's OOM; cheap to sweep, real hardware constraint.
- **Build/CI resource profiling** `[have-now]` — -j, LTO, debug/release, cache state → build
  wall-clock, peak RSS. *Useful:* schedule runners, predict CI duration before pushing.
- **Database query performance** `[have-now]` — indexes, work_mem, table scale → latency, plan
  choice, **spill-to-disk**. *Useful:* predict the plan flip before it happens on prod-size data.
- **Media encoding** (ffmpeg) `[have-now]` — codec, preset, CRF, resolution → encode time, file
  size, VMAF quality. *Useful:* pick the rate/quality/speed point without sweeping.
- **OS/kernel tuning** `[have-now]` — sysctl/scheduler/network-stack knobs → benchmark
  throughput/latency. *Useful:* tuning without reboot-and-measure loops.

### 2. Energy & thermal

- **GPU power capping** `[have-now, H100]` — nvidia-smi power limit × workload → throughput,
  joules/token, clocks. *Useful:* perf-per-watt frontier for datacenter capacity and energy cost
  forecasting, measured on the actual card.
- **Apple Silicon power/thermal** (powermetrics) `[have-now]` — workload config → package watts,
  **thermal-throttle onset**, sustained clock. *Useful:* the only *physically* measured signal
  (heat) we can capture today; predicts sustained vs burst performance.
- **Battery drain / cycle life** `[public-data: NASA, Severson et al.]` — charge protocol,
  temperature, load → capacity fade, **end-of-life**. *Useful:* each real data point costs weeks
  of cycling; the canonical "expensive physical measurement" domain.
- **Datacenter/HVAC cooling** `[acquire]` — setpoints, load placement → PUE, hotspot temps.
  *Useful:* counterfactual cooling policies without risking thermal incidents.

### 3. Accuracy & quality (predicting experiment outcomes)

- **Hyperparameter → final accuracy** `[public-data: LCBench, HPO-B, YAHPO; have-now for small
  models]` — HP config (+ optional early learning-curve prefix) → final val accuracy, wall-clock.
  *Useful:* kill bad runs early, HPO without full training; large public sweep datasets exist to
  distill.
- **ML-experiment outcome prediction** (MLE-bench / PaperBench style) `[synthetic-from-agent-runs,
  expensive]` — an ML-engineering plan or paper-reproduction attempt → achieved score,
  success/failure. *Useful:* route agent compute to attempts likely to work; meta-level estimator
  for agentic ML engineering. Each ground-truth point is itself an expensive agent run — a later
  corpus, possibly distilled from published benchmark run logs.
- **Quantization/pruning → accuracy drop** `[have-now]` — method, bits, layer selection → eval
  score delta, speedup. *Useful:* pick compression without an eval sweep per candidate.
- **Data mix / scale extrapolation** `[public-data + have-now small-scale]` — corpus mix, model
  scale → loss. *Useful:* the scaling-law question every lab asks; hardest generalization claim.
- **RAG/prompt config → eval score** `[have-now — dogfood]` — retriever k, prompt variant, model
  → suite fidelity, cost. *Useful:* wmh predicting its own eval outcomes; recursive but real.

### 4. Reliability, failure & safety

- **Load/chaos testing** `[have-now]` — request rate, concurrency, pod memory limits → p99
  latency, error rate, **collapse point**. *Useful:* find the breaking point without breaking
  prod; counterfactual load tests.
- **Container/K8s sizing** `[have-now]` — cgroup memory/CPU limits × workload → **OOM-kill**,
  throttling, runtime. *Useful:* right-size deployments; real kernel OOM-killer as ground truth.
- **API rate-limit behavior** `[have-now — we already meter this]` — request pattern, model,
  region → throttle rate, retry latency. *Useful:* capacity-plan against providers; the Bedrock
  throttling traces exist as a byproduct of every capture we run.
- **Flaky-test probability** `[have-now]` — test, parallelism, seed, machine load → pass/fail
  distribution. *Useful:* predict flake before merging retry-loops into CI.
- **Hardware-damage envelopes** `[acquire]` — over-voltage/thermal-runaway boundaries.
  *Useful:* the safety case for physical iteration — never measure the cliff by falling off it.

### 5. Cost & time

- **Cloud job cost/duration** `[have-now]` — instance type, spot vs on-demand, job config →
  $ and wall-clock (composite over category 1). *Useful:* quote the run before submitting it;
  finops as a world-model query.
- **LLM API cost/latency** `[have-now — MeteredProvider already records it]` — model, prompt
  shape, batching → tokens, $, latency. *Useful:* free corpus from our own tracking data.

### 6. Search & decision loops — prediction gates an expensive evaluator

Domains where the pre-run estimator's consumer is a search/decision loop, so ranking accuracy and
decision regret matter more than absolute error. (Lit review sections: superoptimization,
databases, broader map.)

- **Kernel/code-optimization search** (KernelBench, STOKE, Souper) `[have-now via released
  artifacts]` — candidate kernel/edit → correct?, speedup bucket. *Useful:* the measurement/
  verification step bottlenecks the whole search; a learned gate cuts solver-bound
  superoptimization ~2× (PrediPrune) and lets kernel search consider several× more candidates
  per GPU budget (GPU Forecasters). GPU Forecasters ships 424 eval pairs + search logs we can
  reuse directly.
- **Query cardinality / cost / what-if index estimation** `[have-now]` — query or hypothetical
  index → cardinality, latency, plan choice. *Useful:* cardinality misestimates cause
  orders-of-magnitude plan slowdowns (up to 10⁸ underestimation; 40% of queries ≥2× slower —
  Leis VLDB'15); what-if index advisors have shipped in production DBMSs since 1998 — the
  25-year-old proof this product category works.
- **LLM routing / cascade prediction** `[have-now — dogfood]` — query → will the cheap model
  suffice? *Useful:* RouteLLM ~85% cost cut at 95% quality on MT-Bench (dataset-dependent;
  enterprise reports 40–70%); pure pre-run estimation with instant, huge market resonance.
- **Scaling-law loss forecasting** `[have-now small-scale]` — (params, tokens, compute) → final
  loss. *Useful:* the purest existing pre-run estimator in ML — labs bet tens of millions per
  frontier run on extrapolation from sub-1% cheap runs; a mis-provisioned recipe (pre-Chinchilla)
  costs an entire wasted run.
- **Predictive test selection / build-failure prediction** `[public precedent]` — diff → which
  tests fail? *Useful:* Meta measured a 2× cut of total testing infra cost at >99.9% faulty-change
  recall; Google runs 150M test executions/day, 99% of which pass. Self-labeling corpus at scale.
- **A/B experiment outcome via surrogates** `[acquire]` — variant + early metrics → long-horizon
  outcome. *Useful:* Netflix saw ~95% consistency predicting 63-day outcomes from 14 days —
  multiplying experiment throughput; known elevated false-positive risk to manage.
- **HPC queue-wait + runtime prediction** `[have-now precedent]` — job config → start time,
  runtime. *Useful:* bad walltime estimates cripple backfill on machines costing tens of
  millions; XGBoost hits 85%-within-60-min on production traces.
- **Spot-eviction / capacity prediction** `[vendor-grade evidence]` — workload placement →
  time-to-eviction. *Useful:* the up-to-90% spot discount is only safely capturable with eviction
  lead-time; asymmetric-cost time-to-event prediction.
- **Network config what-if** `[formal-tool incumbent]` — config change → forwarding behavior,
  outage? *Useful:* an hour of downtime is >$300K for >90% of large enterprises (ITIC); Batfish
  shows the market pays for guaranteed pre-deploy answers — a learned WM competes only on
  heterogeneous breadth.
- **Pre-silicon verification / tape-out risk** `[acquire]` — design state → respin? *Useful:*
  mask sets alone are >$10M at 7nm, ~$40M at 3nm; the extreme end of the value curve where even
  a mediocre estimator pays.
- **Drug/materials screening triage** `[public-data]` — candidate → assay outcome. *Useful:*
  discovery phase ~$430M/5–6 years; ML triage already gates which wet-lab experiments run.

### 7. Physical hardware & manufacturing — the wedge

All `[acquire]` unless noted; each iteration costs days-to-weeks and real money, which is exactly
why a grounded estimator is valuable — and why v1 uses the cheapest-to-measure proxies above to
prove the method first.

- **PCB design** — stackup, trace geometry, materials → impedance, crosstalk, EMC pre-scan
  results, **bring-up failures**. *Useful:* a respin costs weeks + thousands of dollars; board
  bring-up logs are the trace corpus a hardware org already has.
- **Antenna/RF tuning** — geometry, matching network → S11, gain, bandwidth.
- **3D printing** `[semi-have-now: slicer estimates free; real print outcomes need a printer]` —
  layer height, speed, temps → print time, **warp/adhesion failure**.
- **CNC machining** — feeds, speeds, material → surface finish, tool wear, **chatter**.
- **Semiconductor process** `[public-data: SECOM]` — process parameters → yield, defect class.
- **Turbofan/rotating-machinery degradation** `[public-data: NASA C-MAPSS]` — operating profile →
  remaining useful life. *Useful:* the classic prognostics benchmark, reframed as WM estimation.
- **Wet-lab / materials synthesis** — protocol parameters → yield, purity. *Useful:* the
  science-vertical version of the same loop.

## Quantified value anchors (verified 2026-07-10; full sourced table in `.agents/docs/research/ff4-lit-review.md`)

| Domain | The expensive loop | Verified value of prediction |
|---|---|---|
| ML training OOM/throughput (v1) | H100 ~$2–7/GPU-hr; runs = 100s–1000s GPU-hr | 30.7% of cluster jobs fail/killed = ~55% of GPU-time (Philly ATC'19); GPU-OOM = largest DL-specific failure, 8.8% of all failures (ICSE'20) |
| LLM serving config (v1) | Full config sweep for LLaMA2-70B = 42K GPU-hr ≈ $218K | Simulator answers it in 1 CPU-hour (Vidur, MLSys'24, verbatim) |
| FPGA/ASIC P&R (v1) | 30 min–6 hr per FPGA run; ASIC masks $1M (28nm) → ~$40M (3nm) | Respin tail-risk avoidance; iteration-latency compression |
| CI / test selection | 150M test executions/day at Google; 99% pass | Meta: testing infra cost halved at >99.9% faulty-change recall (ICSE-SEIP'19) |
| DB query plans | Bad cardinality → up to 10⁸ misestimates | 40% of queries ≥2× slower on estimates vs truth (Leis VLDB'15); Bao: >50% cloud cost + ~6.5× p99 cut |
| Battery cycle-life | ~6 months per cell to end-of-life | Protocol search 500+ days → 16 days, >30× (Attia, Nature 2020) |
| Kernel/code search | Verification/benchmarking bottlenecks search | ~2× verification cut (PrediPrune); several× more candidates per GPU budget (GPU Forecasters) |
| LLM routing | Frontier-model inference cost | ~85% cost cut at 95% quality on MT-Bench (RouteLLM; dataset-dependent) |
| PCB respins | $10K–$86K + 1–4 weeks per spin (vendor-grade sources — weakest provenance in this set) | ~50% of complex products need ≥1 extra iteration (Aberdeen via Siemens) |

Corrections adopted: battery speedup is >30× (not "~15×"); "65% of DL failures are OOM" is a
denominator conflation — the defensible number is 8.8% of ALL failures (largest DL-specific
category). PCB dollar figures must be labeled vendor-estimates in any launch material.

## Value-per-prediction model (v0)

The anchors above are raw ingredients; to compare use cases we normalize to the **expected value
of one WM prediction** — the blend of direct cost, weighted time, and safety the launch page
should quote per domain:

```
V_query = p_act × [ C_direct  +  λ_t · T_saved  +  p_tail · C_tail ]  −  p_reg · C_reg
```

- **p_act** — probability the prediction changes an action (a run skipped, a config chosen, a
  failure pre-empted). A prediction that confirms what you'd do anyway is worth ~0.
- **C_direct** — direct cost of the real measurement avoided (compute $, materials, tool time).
- **λ_t · T_saved** — loop latency converted to $: engineer-blocked time at λ_t ≈ $100/hr
  (loaded); calendar/product-delay time is worth far more and is flagged per-domain rather than
  folded into λ_t.
- **p_tail · C_tail** — safety/tail term: probability × cost of the catastrophic outcome the
  prediction can pre-empt (respin, outage, hardware damage).
- **p_reg · C_reg** — the regression tax (from the DB literature): probability the estimator
  causes a *worse* action than the default × its cost. This is what kills deployed estimators;
  it must be measured, not assumed zero.

A WM query costs c ≈ $0.001–0.05 (one LLM inference), so **ROI = V_query / c**. Three leverage
modes matter when reading the table: **per-gate** (one prediction gates one run), **per-search**
(thousands of predictions multiply one search's throughput), and **per-policy** (one prediction
sets a fleet-wide operating point — value scales with the fleet, not the query).

| Use case | p_act (est.) | C_direct avoided | Time term | Tail term | **V_query (order)** | Dominant | Mode |
|---|---|---|---|---|---|---|---|
| ml-memory / OOM (v1) | 5–10% of submissions | 1–8 GPU-hr wasted = $2–56 | 0.5–4 hr turnaround = $50–400 | — | **$3–45** | time | per-gate, high volume |
| vllm-serving config (v1) | 70–90% of sweep points | ~140 GPU-hr ≈ $700/config (Vidur ÷ "hundreds") | days of setup per trial | SLO breach (unpriced) | **$300–600** | cost | per-gate / per-search |
| fpga-pnr (v1) | ~50% of candidates | 0.5–6 hr tool+seat = $60–1,500 | engineer wait, same hours | — | **$30–1,000** | time+cost | per-gate |
| Pre-silicon / tape-out risk | rare, decisive | — | months of slip | Δp≈1% × $10–40M masks | **$10⁵ per averted-risk query** | safety | per-gate, low volume |
| Battery cycle-life (bridge) | ~90% of protocol candidates | channel + lab $10³–10⁴ (est., unpublished) | **6 months calendar** per test | thermal events (unpriced) | **$10³–10⁴ + months** | time | per-gate |
| PCB bring-up (bridge) | 10–50% catch rate | respin $10K–86K (vendor-grade) | 1–4 weeks per spin | field failure | **$10³–4×10⁴** | cost+time | per-gate |
| Power/thermal operating point | one decision per fleet/quarter | 10–25% energy × fleet-year (8-GPU node ≈ $30K/yr at 15%) | — | throttle/damage envelope | **$10²–10⁴ per policy** | cost (fleet-scaled) | per-policy |
| Kernel/code search | 50–90% pruned | benchmark/verify $0.01–0.5 | queue seconds | — | **$0.01–0.5** | cost | per-search ×10³–10⁵ |
| LLM routing | ~50% routable | model-cost delta $0.005–0.05/query | latency win (cheap model faster) | quality regression | **$0.003–0.03** | cost | per-gate ×10⁶⁺/day |
| CI test selection | 2/3 of executions | $10⁻³–10⁻² per execution | dev feedback latency | escaped regression = C_reg driver | **$0.001–0.01** | cost | per-gate ×10⁸/day |
| DB plan/cardinality | tail queries only | median ~0; tail 10–1000× slowdown | analyst wait | silent 10⁸ misestimate | **$0.001–0.1 (tail-skewed)** | safety-of-tail | per-gate ×10⁶⁺/day |
| Media encoding ladder | ~80% of ladder points | 10–100 CPU-hr ≈ $0.3–3/encode | none critical | — | **$0.3–3** | cost | per-search |

**How to read it.** Value per prediction spans ~8 orders of magnitude, and the portfolio splits
cleanly: **high-value/low-volume** decisions (tape-out, PCB, battery, serving-config — one good
prediction pays for millions of WM queries) versus **low-value/high-volume** streams (routing,
CI, DB, kernels — value comes from aggregate throughput and the regression tax dominates the
design). Our v1 trio deliberately samples the middle of the curve (per-query $3–$1,000) where
ground truth is cheap enough to *measure* p_act and p_reg honestly rather than assume them —
exactly the two parameters this table currently estimates. The launch page should print this
table with v1's p_act/p_reg cells filled in from our own experiments, and the bridge domains
(battery, PCB, tape-out) quoted as the value gradient the same estimator climbs next.

**Caveats.** All non-anchor cells are order-of-magnitude estimates (assumptions inline);
λ_t=$100/hr is a choice, not a fact; product-delay time (battery's 6 months, tape-out's slip)
is priced qualitatively because published $/week-of-delay numbers don't exist; PCB inputs are
vendor-grade. The model's purpose is ranking and framing, not accounting.

## v1 choice (user, 2026-07-05): start with the simplest three and see how they perform

1. **ml-memory** — torch train/infer sweeps on M3 Max (MPS/CPU); optional CUDA leg on the H100
   dev box. Headline: OOM classification + peak-memory/throughput relative error.
2. **vllm-serving** — vLLM sweeps on the H100 dev box: fits-at-context?, throughput, latency.
3. **fpga-pnr** — yosys/nextpnr iCE40/ECP5 sweeps locally: fits?, LUTs, Fmax.

All three share one corpus schema (config tool-call → measured JSON), one judge
(`wmh.optimize.numeric.NumericJudge`: relative error + threshold classification), and one
baseline table (analytic formula where one exists, k-NN over training sweeps, mean predictor).
Honesty rule: if a baseline wins, the table ships anyway and the framing shifts to provenance +
breadth ("one estimator, any process you have traces for, grounded in your hardware"), not
point accuracy.

## Capture architecture: extend `environment-capture` (PR #56) with sweep capture

PR #56's contract is agent-runs-a-benchmark: `BenchmarkAdapter.tasks/open_env/grade`,
`CommandEnv.execute(command) -> ExecResult`, `run_capture` (per-task failure isolation), an OTel
GenAI JSONL emitter pinned against `wmh.ingest`, and `CommandEnv.execute` as the WM-swap seam.

**Assessment: reuse is the right call.** A sweep session maps cleanly onto the existing record
types — `Task` = one sweep spec ("characterize gpt2-family training memory on MPS"),
`StepRecord` = one (config tool-call → measured-JSON observation) transition, `Trajectory` = the
sweep session, `trajectory_to_spans` emits the corpus with zero wire-format work, and
`run_capture`'s isolation is exactly what a sweep needs (measurement runs *crash by design* —
OOM is data, not an error). Two deliberate deltas, both additive:

1. **The "agent" is not an LLM.** A `SweepPolicy` (grid / random / adaptive-bisection toward the
   cliff) implements the existing `CaptureAgent` protocol. Consequence worth advertising:
   sweep capture is deterministic and token-free — corpus cost is pure measurement time.
2. **Measurement envs run the config in a subprocess.** A generic
   `SubprocessMeasurementEnv(measure_fn)` executes one JSON-config command per fresh child
   process (an OOM must kill the child, never the sweep), returns measured JSON as
   `ExecResult.output`, and encodes *outcome* failures (OOM, doesn't-fit, timing-fail) as fields
   in the measurement — `is_error` stays reserved for "the measurement itself broke". This is
   the general "capture traces from a package" mechanism: a domain dir contributes one pure
   `measure(config) -> measurements` function over its package (torch, vLLM client, yosys
   wrapper) plus a config-space spec; everything else is shared.

Open contract question for WS-B2 (via DECISIONS.md): `grade()` is meaningless for sweeps
(`Trajectory.reward` is already optional). Options: (a) sweeps return `reward=None` through a
trivial `grade`, or (b) a sibling `SweepAdapter` protocol without `grade`, sharing every other
type. We propose (b) — honest surfaces over vacuous methods — but implement whichever WS-B2
prefers, since they own the package.

Layout (follows PR #56's benchmark-dir pattern): shared sweep machinery in
`environment_capture/sweep.py` (+ tests, inside the python gate); domain dirs
`packages/environment-capture/{ml-memory,vllm-serving,fpga-pnr}/` each with `measure.py`,
`configspace.py`, `capture.py`, `README.md`, committed `traces.otel.jsonl`, and `evals/*.toml`;
heavy deps (torch pinned, vllm, yosys via oss-cad-suite) in domain-local venvs, gate-excluded.
Sequencing risk: PR #56 is open — FF4 branches from it (or rebases when it merges) rather than
duplicating the contract; if #56 stalls, fallback is the older self-contained
`examples/<name>/` converter pattern.

## Eval & demo

- **Open-loop**: D12 conventions, scored by `NumericJudge` — per-field relative error +
  threshold classification (OOM/fits) from `JudgeResult.dimensions`. Held-out = random configs
  (interpolation headline) **plus** one held-out region per domain (e.g. batch sizes above
  anything swept) reported separately as the extrapolation stress test.
- **Closed-loop teaser**: `wmh.env.run_episode` with a sweep agent searching for "max batch that
  fits" against WM vs real env — does the WM-guided search land on the same config with N× fewer
  real runs?
- **Demo**: `wmh play` REPL against the built models ("batch 64, seq 2048, fp16, MPS → ?") — the
  ask-your-hardware GIF.
