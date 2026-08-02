# Execution-scored coding-model router protocol

Status: protocol frozen; the original smoke is archived as invalid, the authorized replacement
passed, and the 48-cell fast development tranche is complete. The full sweep was paused on the
user-directed cross-dataset pivot. The follow-up is documented in
`coding-model-router-cross-dataset-20260729.md`. The later Terminal-Bench completion attempt was
stopped when the user explicitly superseded this benchmark direction with DeepSWE-only
optimization. It must not be resumed. The final DeepSWE result is documented in
`coding-model-router-deepswe-20260729.md`.

Experiment ID: `coding-router-20260728`

Source commit: `c3267f1f9d5f35a14ad45b6a94b7b21d3b11c958`

Branch: `exp/coding-model-router-20260729`

Worktree:
`/Users/admin/Documents/experientiallabs/.codex/worktrees/world-model-optimizer/coding-model-router-20260729`

Material paid sweep ceiling: USD 20,000, authorized by the user on 2026-07-29.

## Objective

Build and serve the least expensive pre-inference WMO routing policy that retains at least 95
percent of the held-out quality of the strongest OpenAI or Anthropic static model selected on fit,
while saving at least 40 percent of its inference cost. Provider-reported usage is used when
available; otherwise cost is a clearly labeled trace-derived estimate. Prefer at least 60 percent
savings when the same quality and statistical gates hold.

## Scientific gates

1. Select the static frontier baseline using fit rows only.
2. Held-out relative quality retention must be at least 0.95.
3. Held-out realized inference cost must be at least 40 percent below the baseline.
4. Report relative retention and absolute percentage-point quality delta.
5. Use paired scenario-level 95 percent confidence intervals.
6. No benchmark may lose more than 10 relative quality points or 10 absolute points.
7. Apply the point-estimate gates independently on split seeds 0 through 4.
8. Promotion requires all five seeds to pass plus the pooled paired interval gate.
9. The production choice is the least expensive preregistered point that passes.
10. Held-out results may not change the search space, dial grid, or selection rule.

## Benchmarks

| Cohort | Execution pin | Primary reward | Split group |
| --- | --- | --- | --- |
| Terminal-Bench 2 | Harbor `terminal-bench@2.0`, 89 tasks, `terminal-bench-2` commit `69671fbaac6d67a7ef0dfec016cc38a64ef7a77c` | official Harbor verifier | task family |
| SWE-bench Verified | Harbor `swebench-verified@1.0`, 500 tasks, `harbor-datasets` commit `86723674f04e4209ac479d0fb75d9d9f44b4377e` | official repository tests | repository |

The SWE-bench execution identities are cross-checked one for one against
`princeton-nlp/SWE-bench_Verified` commit
`c104f840cc67f8b6eec6f759ebc8b2693d585d4a`. The frozen parquet has SHA-256
`a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd`.

LiveCodeBench is excluded from the primary protocol because this source revision has no reliable
execution-scored WMO adapter. It may be added only as a separately labeled future cohort.

The aggregate gives each benchmark weight 0.5, regardless of task count. Per-benchmark results
remain primary.

## Splits and leakage controls

Seeds are 0, 1, 2, 3, and 4. A deterministic SHA-256 ordering and subset optimizer chooses whole
groups closest to the 70 percent fit target. Every seed is 62/27 for Terminal-Bench 2 and 350/150
for SWE-bench Verified. All seeds are unique, every task appears exactly once, and no group crosses
fit and held-out.

Router features contain only the task statement and metadata available before the first model
call. Patches, hidden tests, verifier output, reward, future tool trajectory, and held-out labels
are excluded.

### Fast development tranche

The full matrix is scheduled in two resumable stages so implementation feedback does not wait for
SWE-bench setup and repository test latency. `fast-dev` is the first persisted tranche of the real
matrix, not another smoke or a disposable pilot. Every gradeable row is reused unchanged by the
full matrix.

The tranche contains 48 cells: four anchor arms crossed with 12 Terminal-Bench 2 tasks. The arms
are `oai-sol-high`, `oai-luna-high`, `ant-opus5-high`, and `ant-haiku45`, covering both providers
and one frontier and inexpensive arm from each. Candidate tasks are the intersection of the
Terminal-Bench fit assignments across all five outer seeds. They are ordered by SHA-256 of
`fast-dev-v1:<task-id>` and the first 12 are selected:

1. `git-multibranch`
2. `hf-model-inference`
3. `model-extraction-relu-logits`
4. `compile-compcert`
5. `configure-git-webserver`
6. `mteb-leaderboard`
7. `schemelike-metacircular-eval`
8. `winning-avg-corewars`
9. `break-filter-js-from-html`
10. `db-wal-recovery`
11. `financial-document-processor`
12. `extract-moves-from-video`

This tranche may guide implementation and offline router debugging only. It cannot select the
production baseline, lock a promotion configuration, touch any outer-heldout reward, or support a
headline claim. The `full` stage resumes the same matrix and fills every remaining Terminal-Bench
2 and SWE-bench Verified cell before nested fit-only selection begins.

### Superseded Terminal-Bench completion stage

After the cross-dataset effort-router formulation was frozen, `terminal-full` began filling the
remaining Terminal-Bench 2 model-by-task cells. It used the same task manifest, model pool,
attempts, retry policy, scoring, ledger, artifact paths, and `full/outcomes.json`.

The user then explicitly stated that Terminal-Bench was not scientifically interesting and
directed all further optimization to DeepSWE v1.1. The process was interrupted, its partial
artifacts were preserved, and no result from this incomplete stage supports the final claim. It
must not be resumed or expanded. This section remains only as an audit record of the frozen
protocol that was superseded.

Frozen artifact SHA-256 values:

| Artifact | SHA-256 |
| --- | --- |
| Terminal-Bench 2 manifest | `7f59be4fbcc715fcbadf998ccf3d11f00f7fc9c978b196407e0c051c76f6b835` |
| SWE-bench Verified manifest | `6c773fc71e8bda17bb907d2f309c50d3f1b5d887dfda8a6841df066d2f2eed79` |
| Model pool | `b909fe11bc1c89d6e6501d9b750b468aa4abdfab5e389357980608049f99663e` |
| Seed 0 | `83b2a7334a0fa57914b12769d173d78deffb8f84397f18d3accd83a32c6c81e2` |
| Seed 1 | `211a0e1564958691f52a753e1478a58b95acfb4dd2717856bedf6f03a0ca0d3c` |
| Seed 2 | `8d182367772b21d15e59795c7b6eee47c89b39dc77e34520400448121470beb3` |
| Seed 3 | `d8abbbd5d6a451146842b4739c79d024c5f9700ca3286ffcce7ba89c63025716` |
| Seed 4 | `eff42456f5ddd386f31a6b4a4263bd6934bcc3a7637453f0cb90abf87297120b` |

## Candidate roster

| Arm | Exact provider model | Effort | Standard input/cache/output USD per million |
| --- | --- | --- | --- |
| `oai-sol-max` | `gpt-5.6-sol` | max | 5.00/0.50/30.00 |
| `oai-sol-high` | `gpt-5.6-sol` | high | 5.00/0.50/30.00 |
| `oai-terra-max` | `gpt-5.6-terra` | max | 2.50/0.25/15.00 |
| `oai-terra-high` | `gpt-5.6-terra` | high | 2.50/0.25/15.00 |
| `oai-luna-high` | `gpt-5.6-luna` | high | 1.00/0.10/6.00 |
| `oai-gpt55-high` | `gpt-5.5-2026-04-23` | high | 5.00/0.50/30.00 |
| `oai-codex53-high` | `gpt-5.3-codex` | high | 1.75/0.175/14.00 |
| `oai-mini54-high` | `gpt-5.4-mini-2026-03-17` | high | 0.75/0.075/4.50 |
| `ant-fable-max` | `claude-fable-5` | max | 10.00/1.00/50.00 |
| `ant-opus5-max` | `claude-opus-5` | max | 5.00/0.50/25.00 |
| `ant-opus5-high` | `claude-opus-5` | high | 5.00/0.50/25.00 |
| `ant-sonnet5-high` | `claude-sonnet-5` | high | 3.00/0.30/15.00 |
| `ant-sonnet5-low` | `claude-sonnet-5` | low | 3.00/0.30/15.00 |
| `ant-haiku45` | `claude-haiku-4-5-20251001` | off | 1.00/0.10/5.00 |

Each exact ID was returned by its provider's live read-only model-list API on 2026-07-28. Sonnet 5
uses the standard 3/15 price rather than a temporary introductory rate.

Exact model capabilities frozen from the providers' official model pages:

| Exact model | Context | Max output | Structured tool use | Availability at freeze |
| --- | ---: | ---: | --- | --- |
| `gpt-5.6-sol` | 1,050,000 | 128,000 | Responses function calling | live model list |
| `gpt-5.6-terra` | 1,050,000 | 128,000 | Responses function calling | live model list |
| `gpt-5.6-luna` | 1,050,000 | 128,000 | Responses function calling | live model list |
| `gpt-5.5-2026-04-23` | 1,050,000 | 128,000 | Responses function calling | live model list |
| `gpt-5.3-codex` | 400,000 | 128,000 | Responses function calling | live model list |
| `gpt-5.4-mini-2026-03-17` | 400,000 | 128,000 | Responses function calling | live model list |
| `claude-fable-5` | 1,000,000 | 128,000 | Messages tool use | live model list |
| `claude-opus-5` | 1,000,000 | 128,000 | Messages tool use | live model list |
| `claude-sonnet-5` | 1,000,000 | 128,000 | Messages tool use | live model list |
| `claude-haiku-4-5-20251001` | 200,000 | 64,000 | Messages tool use | live model list |

OpenAI publishes a Tier 1 floor of 500 requests/minute and 500,000 tokens/minute for these
roster entries; higher account tiers vary by model. Anthropic's published Start tier is
1,000 requests/minute for every Claude entry, with 500,000 input and 100,000 output
tokens/minute for Fable 5, and 2,000,000 input and 400,000 output tokens/minute for Opus 5,
Sonnet 5, and Haiku 4.5. Account and workspace overrides may be lower. The experiment therefore
starts at four concurrent cells, observes response rate-limit headers, obeys `retry-after`, and
never interprets a pre-execution 429 as a gradeable model failure.

GPT-5.5 and GPT-5.6 calls above 272,000 input tokens are priced per request at the documented
long-context tier: 2x input and cached-input rates plus 1.5x output rates. The runners persist
per-call input, cached-input, cache-write, and output counters because an episode aggregate
cannot determine that tier.

Official sources:

- `https://developers.openai.com/api/docs/models`
- `https://developers.openai.com/api/docs/models/gpt-5.5`
- `https://developers.openai.com/api/docs/models/gpt-5.3-codex`
- `https://developers.openai.com/api/docs/models/gpt-5.4-mini`
- `https://platform.claude.com/docs/en/about-claude/models/overview`
- `https://platform.claude.com/docs/en/api/rate-limits`

## Attempts, retries, and evidence

The primary matrix uses one model attempt per task with the same harness, turn cap, output cap,
wall timeout, and official verifier. Gradeable model failures are never retried. Only a missing
environment, sandbox failure, transport loss before execution, or missing verifier reward is an
infrastructure failure. It receives at most two fresh-sandbox retries after 15 and 60 seconds.
Turn-cap exhaustion, budget exhaustion, no-action or no-tool-call termination, output truncation,
and unparsed tool calls remain gradeable agent failures with the official verifier reward.

Every attempt is retained. Every completed cell atomically persists reward, success, tokens,
cache reads and writes, reasoning tokens, model cost, per-call latency, wall time, tool calls, stop
reason, completion status, failure class, attempt number, and raw Harbor artifact path. Usage and
cost are marked `exact` when provider counters exist. Otherwise, an officially scored cell remains
gradeable and is marked `estimated`: each trace step is treated as one provider call, input tokens
are estimated from the cumulative task plus action-observation transcript with a fixed 4,096-token
system and tool-schema allowance, and output tokens are estimated from the serialized action.
Environment duration is recorded even when E2B does not expose its invoice rate.

During the fast development tranche, Luna's first `db-wal-recovery` attempt exhausted the
900-second agent wall budget after 20 provider calls. Harbor still ran the official verifier and
recorded reward 0. The matrix runner initially hid that grade behind its generic `error`
classification and launched attempt 2. The operator interrupted only that retry, terminated its
exact sandbox, preserved its partial artifacts, and charged a conservative USD 500 budget debit.
The classifier now treats every post-execution failure with an official reward as gradeable while
keeping pre-execution failures retryable. Attempt 1 is the canonical zero; attempt 2 is excluded
from scientific results as a protocol retry.

The fast tranche completed all 48 gradeable cells on 2026-07-29. Official pass counts were 6 of 12
for `oai-sol-high`, 5 of 12 for `ant-opus5-high`, 3 of 12 for `oai-luna-high`, and 1 of 12 for
`ant-haiku45`. Every canonical matrix reward matched both Harbor's raw `result.json` reward and
`verifier/reward.txt`, with exactly one completed canonical ledger event per task and arm.

One Opus `compile-compcert` infrastructure attempt exposed a remote teardown defect: GNU
`timeout` sent TERM at the 900-second boundary, but the Node worker remained alive until the outer
SSH subprocess raised after its 60-second grace, so Harbor could not run the verifier. The remote
command now uses `timeout --kill-after=10`, inside the existing 60-second outer teardown grace.
Post-provider infrastructure attempts with missing counters now retain a labeled trace estimate
instead of incorrect exact zero usage. The migration repaired four historic rows, including the
ungradeable Opus attempt at an estimated USD 0.650145.

The first full-stage launch exposed a separate local concurrency defect before provider execution.
Four independent Harbor scorers created local SSH Pi runtimes simultaneously, but each runtime
defaulted to shim port 8891 and remote directory `ep-8891`. Three cells failed immediately with
`Address already in use`; their exact-zero infrastructure attempts remain preserved and retryable.
The fourth experiment-owned sandbox was terminated after the runner was interrupted. PiRuntime now
binds an OS-assigned local port by default and derives a distinct remote episode directory from
that port. A parallel regression test holds two default runtimes open concurrently and proves both
ports and workdirs differ.

The 32-way Terminal-Bench completion launch exposed a second pre-provider concurrency defect.
Each cell resolved the same remote Harbor dataset into one shared git cache, and concurrent
`git checkout` operations raced. One `video-processing` reservation has no outcome because scorer
construction failed before provider execution. Remote task pinning now takes an async
cross-process file lock around Harbor's shared checkout refresh. The resume uses
`--recover-stale-reservations`; this preserves an unmatched reservation as an interrupted
unknown-cost event with the full USD 500 conservative debit, then frees the deterministic event id
for the real retry.

Agent and verifier timeouts are now separate. The agent remains bounded at 900 seconds, while the
official verifier gets 2,700 seconds. This is required for
`torch-tensor-parallelism`, whose verifier installs PyTorch before running the official tests and
exhausted the earlier 900-second verifier limit. A cell that reaches its three paid-attempt limit
may be recovered only by replaying the preserved tool actions into the exact task with
`ReplayWmoTraceAgent`, making zero model calls, and then running the official verifier. The paid
attempt, replay trial, both replay-agent sources, task checksum, official reward, and content
digests are combined under a separate recovered artifact with explicit provenance.

One metered OpenAI `max_output_tokens` failure also revealed that Harbor can label an
agent-side provider truncation as infrastructure when no verifier runs. A provider-error trace
with recorded model execution is now a definite agent-failure zero. The earliest such attempt is
canonical and any later retry is excluded, so provider truncation cannot benefit from
infrastructure retry selection.

The first hardened resume exposed temporary budget backpressure rather than real ceiling
exhaustion. Thirty-one concurrent cells held conservative USD 500 reservations, so the
thirty-second reservation could not fit even though completed cells would release nearly all of
that headroom. The scheduler now defers the blocked cell and refills the queue only after an
in-flight completion releases a reservation. It stops only when a reservation is blocked with no
work in flight, which proves that no future completion can create headroom.

## Single smoke gate

The only paid pre-sweep gate has exactly four cells:

- fit task `break-filter-js-from-html`;
- held-out task `log-summary-date-ranges`;
- `oai-luna-high`;
- `ant-haiku45`.

Each cell runs separately through Harbor with an E2B task environment and official verifier, then
persists before the next cell starts. After two cells the runner intentionally exits and is
resumed. The resumed run must retain byte-identical completed artifacts, finish the other two
cells, fit a guarded hashing-1024 kNN plumbing policy on the fit task, and replay the held-out task.
The smoke has a hard USD 10 inference cap and is not headline evidence.

The smoke and matrix runners fail before paid execution when the read-only E2B listing shows no
slot under the configured 1,000-sandbox account cap. The user confirmed the expanded account
capacity on 2026-07-29 before the replacement smoke launch.

The no-spend task and provider preflight passed for all four cells. Shared E2B capacity later
opened to 82 active sandboxes, so the one authorized smoke was launched.

The four first attempts failed before provider execution because the configured Pi SSH runner had
no accepted host key. They are preserved as known-zero infrastructure failures. Harbor then
resumed two identical deterministic job directories without creating a sandbox or provider
request. Those no-ops are excluded from the canonical attempt count and retained in
`smoke/retry-noops.json`.

Fresh attempt directories reached real model execution for two OpenAI cells and one Anthropic
cell. The local Pi transport did not persist `worker_usage`, so their token counts and exact model
cost cannot be reconstructed from the raw artifacts. The canonical outcome matrix marks all three
as ungradeable `metering_failure` rows and the spend ledger records `model_cost_usd: null`.
Derived fit, replay, and policy artifacts from the earlier incorrect zero-cost interpretation are
quarantined by content digest and the smoke root carries `invalidated.json`.

The transport now attempts to meter every request with per-call input, cached-input, cache-write,
reasoning, output, and latency counters, and preserves partial usage on failure. On 2026-07-29 the
user explicitly removed exact metering as a launch gate and stated that they monitor account usage
externally. Missing counters therefore trigger the labeled trace estimate above rather than making
an official verifier result ungradeable.

This was the single original smoke attempt permitted by the frozen protocol. It did not pass. On
2026-07-29 the user authorized one replacement smoke and a USD 20,000 hard experiment ceiling.
The three historic paid rows remain explicitly unknown-cost. A conservative USD 300 debit is
charged against the ceiling without relabeling it as realized spend. The original smoke tree is
preserved by digest before a fresh replacement root is created.

## Router search

The production family is WMO guarded kNN. The frozen grid covers:

- hashing-1024 and direct OpenAI `text-embedding-3-large` at native 3072 dimensions;
- neighbors 8, 16, 32, and 50;
- relative similarity threshold 0.90, 0.95, and 0.98;
- novelty quantile 0, 0.05, 0.20, and 0.50;
- guard z 0, 0.5, 1.0, and 1.645;
- minimum paired evidence 3, 5, 8, and 12;
- standard-error floor off and on;
- asymmetric cost guard off and on;
- cost-quality dial 0, 0.10, 0.25, 0.50, 0.75, and 1.0;
- benchmark-stratified bank off and on;
- missing-cell minimum coverage 0.8 and 1.0.

Baselines are every static arm, fit-selected best single, cheapest single, seeded random, cost
only, unguarded kNN, guarded kNN, rank routing, and oracle per-task routing. Cascades and retry
escalation are research-only policies and cannot be the production choice.

The analysis also freezes three one-at-a-time, non-production ablations before any outer-heldout
reward is read:

- benchmark-stratified guarded banks, fitted independently on each benchmark's outer-fit rows;
- deterministic 80 percent nonbaseline fit-cell coverage against the dense 100 percent control,
  with the pinned baseline left complete so the statistical guard remains defined;
- a latency-only static model selected by fit p50 latency.

Prompt-cache-aware switching and conversation affinity require a prior conversation incumbent, so
the one-shot benchmark matrix cannot identify their effect. They remain serving-only operational
checks and cannot support the headline quality claim. The latency-only point can diagnose a
speed-quality tradeoff but cannot replace the quality-cost-selected production policy.

### Declared capability slices

Capability reporting uses only scenario identity and task text available before model execution.
Slices overlap and never change the frozen 0.5/0.5 headline benchmark weights:

- every SWE-bench Verified task is `repository-level-bug-fixing` and
  `debugging-and-test-repair`;
- every Terminal-Bench 2 task is `terminal-operation-and-tool-use`;
- `long-context` is the top quartile of frozen pre-call task-text length;
- deterministic keyword rules declare `build-and-dependency`,
  `code-generation-and-translation`, `data-ml-and-scientific`,
  `debugging-and-test-repair`, and `security-and-recovery`.

The exact keyword tuples live in `coding_model_router_analyze.py`. The final result reports the
guarded router's quality, retention, absolute quality delta, cost, savings, completion, latency,
and model mix for each slice, plus how many outer seeds contain that slice.

### Fit-only selection and heldout lock

Hyperparameters are selected without touching any outer heldout reward:

1. Within each outer seed's fit partition, assign whole repositories and task families to five
   inner folds by SHA-256 of
   `inner-v1:<outer-seed>:<benchmark>:<group>`. The same group never appears in inner train and
   validation.
2. Select the static frontier baseline separately on each outer fit partition using the declared
   0.5/0.5 benchmark aggregate. Ties go to lower realized fit cost, then frozen pool order.
3. Start from hashing-1024, 50 neighbors, threshold 0.95, novelty quantile 0.05, z 0.5,
   minimum eight pairs, standard-error floor on, symmetric guard, and dial 0.25.
   In this factorial search, the dial coordinate contributes its native WMO `pick_lam` cost
   pressure while the separately searched novelty, z, and guard coordinates remain authoritative.
   This avoids silently overwriting three earlier coordinates when the dial is visited. The
   deployable artifact records both the dial label and the effective primitive knobs. Serving
   verification separately exercises WMO's standard live dial mapping.
4. Run two deterministic coordinate passes in this order: embedder, neighbors, similarity
   threshold, novelty quantile, guard z, minimum pairs, standard-error floor, asymmetric guard,
   cost-quality dial. Run this search independently inside each outer seed. At each coordinate,
   evaluate every preregistered value by that seed's five-fold inner validation only. A seed's
   configuration may never use another seed's fit rows because those rows can overlap its own
   outer heldout set.
5. A coordinate value is feasible only when that outer seed's inner-validation aggregate retains
   at least 95 percent of its fit-selected baseline and passes both per-benchmark
   catastrophic-regression limits. For each of Terminal-Bench 2 and SWE-bench Verified, quality
   retention must be at least 0.90 and absolute quality loss must be no more than 0.10. Pick the
   lowest-cost feasible value. If none is feasible, maximize retention, then mean quality, then
   lower cost, then the frozen value order.
6. Benchmark-stratified banks and 0.8 missing-cell coverage are reported as one-at-a-time
   ablations from the selected point. They cannot replace the production point unless they were
   selected through the same fit-only rule.
7. Atomically write `selection-lock.json` with five independently selected hyperparameter sets,
   five fit-selected baselines, inner-validation metrics, matrix digest, split digests, code
   commit, and a deterministic deployment consensus before any outer-heldout policy replay. For
   each discrete coordinate, the consensus is the modal selected value; ties use the frozen value
   order.
8. Fit one outer policy per seed using that seed's locked configuration and full fit partition,
   then evaluate that seed's outer heldout exactly once. All preregistered static, dial, guard,
   rank, random, cost-only, and oracle points may be replayed for the locked Pareto report, but no
   heldout result may revise any seed configuration or the deployment consensus.
9. The deployable artifact refits the pre-heldout consensus hyperparameters on all real rows and
   pins the fit-only consensus baseline: majority of the five outer fit selections, with ties
   resolved by mean outer-fit quality, mean outer-fit cost, then frozen pool order. The headline
   heldout claim is the nested five-seed procedure, not a post-hoc evaluation of this all-row
   deployment refit.

## Statistics and world-model comparison

Confidence intervals use 10,000 paired scenario bootstrap resamples respecting repository and task
family clusters. Each resample preserves the 0.5/0.5 benchmark weighting. Promotion requires the
lower bound of the pooled paired retention interval to remain at or above 0.95, in addition to
all five split point-estimate gates. Reports include quality, cost, effective cost per success,
completion, gradeability, latency p50 and p95, model mix, route-away share, guard reversion,
novelty abstention, and declared capability slices.

Quality and completion claims always come from official benchmark verifiers. Cost claims report
exact and estimated portions separately. A cost-savings result with any estimated portion is an
approximate operational comparison, not exact billing evidence.

After real matrices and splits are immutable, Azure GPT-5.5 world-model inference builds a separate
simulated matrix. Real and simulated rows are never pooled. Compare cell agreement, false positive
and negative rates, calibration, model rank, best-single choice, routed-model choice, guard
decision, predicted deltas, and final promotion decision.

The simulated environment is built from exactly one reward-free trajectory from the real
fit-selected deployment-consensus baseline for each task. Rewards and verifier labels are removed,
but task-specific observations remain retrievable, so the comparison measures reconstruction and
decision agreement rather than unseen-task generalization. Candidate actions in simulation use
WMO's native `LLMAgent`; the real matrix uses Harbor with the default Pi agent. That scaffold
difference is an explicit simulation-to-real confound and must be carried into the report.

## Serving gate

Mount the selected policy and evidence bank through `wmo serve`. Exercise unseen coding requests
through the OpenAI-compatible endpoint and verify both provider routes, multi-turn tool calls,
audit evidence, cost and cache accounting, conversation affinity, safe fallback, and the live
cost-quality dial. The bounded check makes eight provider requests: two for an OpenAI tool-call
round trip, one Anthropic path probe, one selected-policy request, one forced-novelty fallback,
one live-dial request, and two cache-aware kNN turns. The second cache-aware turn must persist a
positive incumbent cache credit in the routed-model evidence.

## Spend and durability

The USD 20,000 ceiling is recorded in the freeze summary. The user monitors provider usage
externally; the local ledger remains a rough guard using exact counters when available and labeled
trace estimates otherwise. It carries USD 1,300 in conservative unknown-cost debits: USD 300 for
the archived invalid smoke, USD 500 for the interrupted Sol cell, and USD 500 for the interrupted
Luna protocol retry. The original and one authorized replacement four-cell smoke are the only
paid work permitted before a valid smoke gate.

Raw artifacts live under `.wmo/experiments/coding-router-20260728/` and stay out of Git. The
protocol and one-off runners live under `.agents/`. Long jobs use tmux with persistent logs. A
launch is accepted only after completed-cell counters advance on two successive polls.

## Exact non-secret commands

Run from the isolated worktree root. Paid phases load provider configuration without printing from
`/Users/admin/Documents/experientiallabs/coding-router/.env.local` and E2B configuration from
`/Users/admin/Documents/experientiallabs/platform/.env.local`.

```bash
uv run python .agents/scripts/coding_model_router_freeze.py \
  --out-dir .wmo/experiments/coding-router-20260728
uv run python .agents/scripts/coding_model_router_matrix.py \
  --root .wmo/experiments/coding-router-20260728 --preflight
uv run python .agents/scripts/coding_model_router_smoke.py \
  --root .wmo/experiments/coding-router-20260728/smoke --preflight
```

An explicitly authorized replacement smoke must demonstrate an actual interruption and resume:

```bash
uv run python .agents/scripts/coding_model_router_authorize.py \
  --root .wmo/experiments/coding-router-20260728 \
  --ceiling-usd 20000 --unknown-cost-budget-debit-usd 300
uv run python .agents/scripts/coding_model_router_smoke.py \
  --root .wmo/experiments/coding-router-20260728/smoke \
  --interrupt-after-cells 2
uv run python .agents/scripts/coding_model_router_smoke.py \
  --root .wmo/experiments/coding-router-20260728/smoke
```

After both a valid smoke and a positive authorized ceiling are frozen:

```bash
uv run python .agents/scripts/coding_model_router_matrix.py \
  --root .wmo/experiments/coding-router-20260728 \
  --stage fast-dev --concurrency 4 --timeout-s 900 --verifier-timeout-s 2700
uv run python .agents/scripts/coding_model_router_analyze.py \
  --root .wmo/experiments/coding-router-20260728 develop
uv run python .agents/scripts/coding_model_router_matrix.py \
  --root .wmo/experiments/coding-router-20260728 \
  --stage terminal-full --concurrency 32 --timeout-s 900 --verifier-timeout-s 2700
uv run python .agents/scripts/coding_model_router_matrix.py \
  --root .wmo/experiments/coding-router-20260728 \
  --stage full --concurrency 4 --timeout-s 900 --verifier-timeout-s 2700
uv run python .agents/scripts/coding_model_router_analyze.py \
  --root .wmo/experiments/coding-router-20260728 validate
uv run python .agents/scripts/coding_model_router_embeddings.py \
  --root .wmo/experiments/coding-router-20260728
uv run python .agents/scripts/coding_model_router_analyze.py \
  --root .wmo/experiments/coding-router-20260728 select
uv run python .agents/scripts/coding_model_router_analyze.py \
  --root .wmo/experiments/coding-router-20260728 evaluate
```

The world-model and serving phases remain separate paid gates:

```bash
uv run python .agents/scripts/coding_model_router_world_model.py \
  --root .wmo/experiments/coding-router-20260728 prepare
uv run python .agents/scripts/coding_model_router_world_model.py \
  --root .wmo/experiments/coding-router-20260728 build
uv run python .agents/scripts/coding_model_router_world_model.py \
  --root .wmo/experiments/coding-router-20260728 simulate
uv run python .agents/scripts/coding_model_router_world_model.py \
  --root .wmo/experiments/coding-router-20260728 analyze
uv run python .agents/scripts/coding_model_router_world_model.py \
  --root .wmo/experiments/coding-router-20260728 compare
uv run python .agents/scripts/coding_model_router_serve_verify.py \
  --root .wmo/experiments/coding-router-20260728 prepare
uv run python .agents/scripts/coding_model_router_serve_verify.py \
  --root .wmo/experiments/coding-router-20260728 --port 8765 run
uv run python .agents/scripts/coding_model_router_report.py \
  --root .wmo/experiments/coding-router-20260728
```

Audit the persisted evidence at any point without network access or paid calls:

```bash
uv run python .agents/scripts/coding_model_router_audit.py \
  --root .wmo/experiments/coding-router-20260728
uv run python .agents/scripts/coding_model_router_audit.py \
  --root .wmo/experiments/coding-router-20260728 --require-complete
```

The first command atomically writes `completion-audit.json`. The second also exits nonzero unless
every terminal requirement in the original brief is independently evidenced. A complete audit
accepts either a measured promotion or a measured target-not-reached conclusion, but never treats
missing paid evidence as a negative scientific result.

The later external-trace DeepSWE work, native linear policy, corrected source weighting, and
matched task-blind control are recorded in
`coding-model-router-external-autoresearch-20260730.md`. That control is now required before a
task router can be called a gain over effort mixing.

The current-model BigCodeBench reasoning-effort study is recorded in
`coding-model-router-bigcodebench-effort-20260731.md`. Its corrected held-out-attempt oracle
failed the frozen headroom gate, so no BigCodeBench router was fit. The next external-only study
is preregistered in `coding-model-router-swe-smith-broad-20260731.md`. DeepSWE remains sealed
until that study earns a positive external promotion.
