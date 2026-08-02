# DeepSWE model and reasoning-effort optimizer

Status: complete post-hoc repeated repository-grouped evaluation. The WMO optimizer and serving
artifact work. The selected deployable policy is static because every eligible task-text kNN
candidate is dominated by the static model-effort frontier.

Date: 2026-07-29

## Question

Can WMO select a model plus reasoning-effort policy on DeepSWE v1.1 that retains at least 95
percent of the fit-selected strongest static arm's held-out graded quality while saving at least
40 percent of its measured inference cost?

This follows the separate cross-dataset experiment in
`coding-model-router-cross-dataset-20260729.md`. That experiment trained on
`nebius/SWE-agent-trajectories` and evaluated on DeepSWE. Its exploratory low/high effort router
retained 99.34 percent of static-high quality and saved 37.16 percent, but missed the 40 percent
promotion gate. The strict preregistered transfer preserved quality and increased cost because an
absolute embedding-similarity floor did not transfer across corpora.

## Data contract

- Arm identity is provider model plus reasoning effort.
- The WMO-native scope contains all 41 OpenAI and Claude model-effort arms in the published
  DeepSWE table. The nine other published arms use provider backends WMO does not serve and are
  outside this deployable-policy comparison.
- Reward is computed directly as `f2p_passed / f2p_total`, then averaged across published
  included trials for a task-arm cell.
- Cost is DeepSWE's measured `cost_usd`, averaged by task-arm cell.
- Every pool entry keeps the arm handle in `name`, the provider runtime model in `model`, the
  reasoning effort, and frozen provider input, cache, output, and cache-write rates.
- A task is dropped across all arms if any reward or cost cell is absent. No arm is dropped for a
  sparse cell.
- Three tasks are dropped:
  `ofetch-per-origin-circuit-breaker`, `pwntools-tube-multiplexing`, and
  `superjson-error-stack-serialization`.
- The resulting matrix is 41 arms by 110 tasks from 88 repositories.

Source hashes:

| Artifact | SHA-256 |
| --- | --- |
| `data/deepswe/trials.json` | `7844056bade4cee4a2c2964c9582bf7eb1344735a28695cae7d419055656417a` |
| `data/deepswe/tasks.json` | `b0d25ec0e566c0391e4385a63343b92d5371b67f052e1c9062c9d226d9d18dd1` |
| `results/deepswe_embeddings.json` | `caa21f5a63abfe0eb8f5c545b4304fc29d340e661cdfd2a7b8fdc962129b3968` |

## Selection

Five outer splits use seeds 11, 23, 37, 41, and 59 with 70 percent of repositories on fit.
Repository overlap is asserted empty. Each outer fit set gets a five-fold repository-grouped
nested search.

The search compares static relative-quality floors from 0.95 through 1.00 against WMO guarded
kNN using the same floors, asymmetric guard, `knn_z` in 0, 0.5, 1, and 2, and `pick_lam` in 0,
0.01, 0.02, and 0.03. The absolute novelty floor is disabled. `min_pairs=8` and the small-sample
standard-error floor remain enabled.

A candidate is eligible only if aggregate inner-fold quality retention and every individual
inner-fold retention are at least 0.95. The least-cost eligible candidate wins. This rule selects
a static policy on every outer split.

On the all-task nested fit:

| Candidate | Inner quality retained | Minimum fold retained | Inner savings |
| --- | ---: | ---: | ---: |
| Static, quality floor 0.97 | 0.98975 | 0.96354 | 77.77% |
| Best eligible kNN, floor 0.97, z 2, lambda 0.03 | 1.00445 | 0.97993 | 63.18% |

The kNN candidate retains slightly more quality, but costs substantially more. It is not on the
selected cost-quality frontier.

## Outer held-out result

| Seed | Fit-selected baseline | Selected policy | Quality retained | Quality delta | Savings |
| ---: | --- | --- | ---: | ---: | ---: |
| 11 | Sol xhigh | Terra high | 0.98396 | -0.01462 | 74.50% |
| 23 | Opus 5 high | Luna xhigh | 0.96216 | -0.03468 | 75.57% |
| 37 | Luna max | Luna xhigh | 0.98554 | -0.01308 | 44.93% |
| 41 | Fable 5 xhigh | Luna xhigh | 1.00553 | +0.00522 | 86.43% |
| 59 | GPT-5.5 xhigh | Luna xhigh | 0.98643 | -0.01215 | 80.94% |

Mean held-out quality retention is 0.98472. Mean quality delta is -0.01386 graded reward. Mean
cost savings is 72.47 percent. All five point-estimate gates pass.

Repository-cluster bootstrap support is weaker. The lower 95 percent savings bound exceeds 40
percent on all five seeds, but the lower quality-retention bound exceeds 95 percent on only seeds
37 and 41. The other three quality intervals are statistically inconclusive. Promotion therefore
remains false.

## Full-matrix context

The strongest full-matrix static arm is Opus 5 high at 0.95433 graded quality and USD 679.86.
Luna max reaches 0.94563 at USD 348.42. Luna xhigh reaches 0.92802 at USD 171.01. This is why a
cost-quality optimizer must compare dynamic routing with the static model-effort frontier rather
than only with an expensive strongest arm.

The production nested fit selects `mini_swe_agent_gpt_5_6_luna_xhigh`. Its provider snapshot is:

- kind: `openai_responses`
- runtime model: `gpt-5.6-luna`
- reasoning effort: `xhigh`
- list price per million tokens: USD 1.00 input, USD 0.10 cached input, USD 6.00 output, and
  USD 1.25 cache write

## Serving proof

The persisted `policy.json` was loaded through `RoutingPolicy.load`, mounted as an
`EndpointRuntime`, and called through WMO's OpenAI-compatible FastAPI route. A credential-free
probe provider prevented a paid model request while exercising the real serving selection and
provider-config construction paths.

The HTTP response was 200, `x-wmo-routed-model` named the Luna xhigh arm, and the constructed
provider config retained runtime model `gpt-5.6-luna` plus reasoning effort `xhigh`. The serving
proof passed with zero paid calls.

Artifact hashes:

| Artifact | SHA-256 |
| --- | --- |
| `matrix.json` | `2988742e48b1c9bfec8dc45d88af112c46c45367529d1936b709e4b4e549835f` |
| `policy.json` | `e90aeb861807659672109dbff072fd76fcb3be78eedfc01ce178bc1bc64d5734` |
| `report.json` | `401328e1ca69b40b0365bae5e173157b0ffbe4e213dd1ae1c21f69c2f9787158` |
| `serving-verification.json` | `1f3d1575725db13682fc6881cf2ef010ca3d3df45f9003fe29b1dc2f72e00cbb` |
| `serving-requests.jsonl` | `c67b94049c1976494fe792f2cbe89edd05e630913e1a1c26b3f0523bf79d96c5` |

Ignored artifact root:
`.wmo/experiments/coding-router-deepswe-20260729`

## Interpretation

WMO can optimize DeepSWE and serve the selected model-effort arm. The honest production result is
not a task-text router. It is a static frontier policy, because the available task-text signal
does not justify kNN's additional spend.

The external-trace binary effort router remains a positive research result but not a promoted
policy. The next dynamic-policy experiment should use external tasks with repeated verified
outcomes for multiple reasoning efforts on the same task. More single-effort traces mainly sharpen
the static choice because they do not identify the counterfactual value of effort.

This analysis is post hoc. DeepSWE outcomes were inspected before the repeated grouped protocol
was finalized. It is reproducible evidence, not a fresh confirmatory result.

## Reproduction

Runner:
`.agents/scripts/coding_model_router_deepswe.py`

Tests:
`.agents/scripts/coding_model_router_deepswe_test.py`

The final run made no model or embedding calls and spent USD 0.

Follow-up external-only autoresearch and the native WMO policy are documented in
`coding-model-router-external-autoresearch-20260730.md`. The matched task-blind control confirms
that the current native text router does not add task-selection value beyond effort mixing.
