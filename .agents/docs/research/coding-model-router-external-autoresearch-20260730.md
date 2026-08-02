# External coding trace router autoresearch

Date: 2026-07-30

Status: lane stopped; external Open-SWE signal established, DeepSWE routing value not established

## Objective

Improve DeepSWE v1.1 cost efficiency without fitting any parameter, feature transform, or
threshold on DeepSWE outcomes. The routing action is model plus reasoning effort. The deployed
router must run before inference without another language model call or feedback loop.

## Authoritative ceiling correction

The DeepSWE parent lane retracted the naive same-attempt per-task oracle after this work began.
The corrected held-out oracle chooses each task's arm on two attempts and scores it on two
different attempts over 107 tasks and nine arms. It reports:

| Quantity | Graded reward | 95 percent interval |
| --- | ---: | --- |
| Held-out per-task oracle | 0.9598 | [0.9458, 0.9712] |
| Best static arm | 0.9480 | not applicable |
| Honest headroom | +1.18 points | [-0.52, +3.28] points |

The headroom interval includes zero. DeepSWE 1.1 is too saturated to justify another routing
feature family or another target replay. The static-frontier pass labels below are retained only
as historical audit results. They are not promotion evidence after this correction.

No further DeepSWE feature search, threshold tuning, cost-penalty tuning, or target replay is
permitted in this lane. A future router experiment must first establish materially positive
static-to-oracle headroom on a non-saturated benchmark or expert pool, using held-out attempts.

## External fit protocol

The first frozen fit used 4,484 deduplicated tasks from four sources:

| Source | Raw tasks | Fit tasks | Repository groups | Weak reward | Strong reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nebius SWE Agent 8B and 70B | 527 | 527 | 375 | 0.1133 | 0.1368 |
| R2E Gym GPT-5 Codex and Kimi 2.5 | 3,393 | 3,393 | 10 | 0.4960 | 0.7135 |
| SWE-bench Verified GPT-5.2 effort IRT | 500 | 500 | 12 | 0.6900 | 0.7180 |
| CodeRouterBench OOD176 | 176 | 64 | 22 | 0.3295 | 0.6420 |

The fitter used repository-grouped five-fold cross-validation, source-balanced operating-point
constraints, a task-blind control, a within-source shuffled-label control, and leave-source-out
checks for the two main Ridge candidates. It froze one candidate and the external quality floors
0.95, 0.97, and 0.99 before opening the target artifact.

The selected scorer was `word128-ridge-heads-a1`. Its external out-of-fold uplift Spearman was
0.0657. At the external 0.95 quality floor it sent 49.83 percent of source-balanced traffic to
the strong arm, retained 96.20 percent mean quality, and retained at least 90.02 percent on every
source.

## Frozen DeepSWE checkpoint

The target replay used 110 fully scored tasks across 88 repositories and graded
`f2p_passed / f2p_total` reward. Static comparisons span 41 model and reasoning-effort arms.
Repository bootstrap resamples repositories, not individual tasks.

The least-cost preregistered point that passed the original static-frontier rule was:

| Field | Result |
| --- | ---: |
| Ladder | GPT-5.6 Luna xhigh to max |
| External operating point | 0.97 |
| Target traffic | 26 xhigh, 84 max |
| Router quality | 0.9392186960 |
| Best static quality | 0.9543283155 |
| Quality retention | 98.4167 percent |
| Router cost | USD 316.1660 |
| Best static cost | USD 679.8643 |
| Cost savings | 53.4957 percent |
| Paired quality delta 95 percent CI | [-0.0437750, 0.0091836] |
| Allowed quality delta | -0.0477164 |
| Static dominated | no |
| Original static-frontier gate | pass, now audit-only |

The external 0.99 point also passed that legacy rule at 99.0667 percent quality retention and
50.1149 percent cost savings. The selection rule chose the cheaper passing point.

This is deployment calibration, not untouched confirmation. The ladder came from previously
known DeepSWE aggregate arm results, even though target task labels did not enter the fit or
thresholds. The authoritative ceiling correction above supersedes the promotion interpretation.

## Weighting audit correction

The first fit exposed a data-combination bug after the target replay. CodeRouterBench has 176
unique prompts internally, but 112 are exact normalized-text duplicates of tasks already loaded
from earlier sources. Keeping 64 rows is the correct cross-source deduplication result. The bug
was that weights were assigned before deduplication, so CodeRouterBench received 64/176 of the
intended source weight while the other sources each received their full weight.

This is not target leakage, but it makes the recorded `equal_total_weight_per_source` claim false
for the first checkpoint. Commit `20da7f5e` changes deduplication to exact normalized text,
retains distinct prompt variants with collision-safe ids, and computes weights after
deduplication. Commit `caf60b81` records per-source weight totals in the fit artifact. The
corrected artifact reports 1,121 total weight for each of the four retained sources.

## Native serving policy

The original selected joblib is 48,671,892 bytes because it carries a learned vocabulary and SVD
transform. It is a research artifact, not a serving requirement.

The native policy uses WMO's deterministic signed character-trigram hashing embedder plus two
Ridge potential-outcome heads. The artifact stores only plain JSON weights, biases, the frozen
threshold, and the two model plus reasoning-effort pool entries. Serve time performs one local
embedding and two dot products. It makes no network call and loads no pickle.

`RoutingPolicy(kind="linear")`, shared offline replay, save and load, validation, sticky routing,
Pareto reporting, and the OpenAI-compatible chat endpoint are covered by tests. The adaptive
native search uses only external outcomes, but the family was chosen after the first DeepSWE
result and must not be reported as untouched target confirmation.

The corrected native external fit selected `hash512-ridge-heads-a1`. Source weights are equal
after deduplication. Its external out-of-fold uplift Spearman is -0.0157, which does not support a
general task-text uplift signal.

The external 0.97 point still passes the original static-frontier promotion gate:

| Field | Corrected native result |
| --- | ---: |
| Target traffic | 40 xhigh, 70 max |
| Router quality | 0.9382796991 |
| Quality retention vs best static | 98.3183 percent |
| Router cost | USD 295.3985 |
| Cost savings vs best static | 56.5504 percent |
| Paired quality delta 95 percent CI | [-0.0442314, 0.0067431] |
| Static dominated | no |
| Original promotion gate | pass |

That gate is insufficient. A 10,000-sample task-blind control randomly assigns the same 70 tasks
to max effort. Its expected quality is 0.9392274 and expected cost is USD 283.9068. The router is
0.0009477 lower quality and USD 11.4917 more expensive than that mean. Its quality is at the
39.58th percentile and its cost is at the 89.71st percentile of matched random assignments.

Conclusion: the native artifact works and reproduces the fitted scorer exactly, but the measured
benefit is effort mixing, not learned task routing. It must not be promoted as a routing
algorithm. The next algorithm must beat a matched task-blind mixture, not only discrete static
arms.

The same control rejects the original word plus SVD router. At its selected external 0.97 point,
the router sends 84 of 110 tasks to max effort. A 10,000-sample task-blind control with the same
traffic has expected quality 0.9414687 and expected cost USD 306.4853. The router is 0.0022500
lower quality and USD 9.6808 more expensive than the random mean. Its quality is at the 23.26th
percentile and its cost is at the 88.92nd percentile. The 0.99 point is statistically
indistinguishable from the matched mixture. Both tested lexical feature families therefore fail
to establish task-selection value.

## Open-SWE external transfer

The next source uses
[`nvidia/Open-SWE-Traces`](https://huggingface.co/datasets/nvidia/Open-SWE-Traces) outcomes joined
by external task id to issue text from
[`nebius/SWE-rebench-V2`](https://huggingface.co/datasets/nebius/SWE-rebench-V2). A projected
parquet scan reads identity and outcome columns instead of downloading the 18.3 GB trajectory
payload. The compact paired dataset stays in E2B.

Within each agent scaffold, the preparation rule selects the two model modes with the largest
external mean-reward gap. It selected OpenHands with Qwen3.5-122B as weak and MiniMax-M2.5
thinking as strong. After joining text there are 14,504 paired tasks across 2,251 repositories
and nine languages. Their mean task rewards are 0.3230 and 0.3957. The fit receives no DeepSWE
path.

The first family uses 27 deterministic issue-shape features. Its IRT variants predict task
easiness, calibrate weak and strong ability offsets, and assign stronger effort to the
middle-difficulty band. The IRT hypothesis failed externally: every observed IRT variant had
negative out-of-fold uplift Spearman. The preregistered family instead selected a two-head Ridge
baseline with 0.0529 out-of-fold uplift Spearman.

The selected external 0.97 point is directionally better than its matched task-blind DeepSWE
control but is not significant:

| Field | Structural two-head result |
| --- | ---: |
| Target traffic | 26 xhigh, 84 max |
| Router quality | 0.9430970943 |
| Router cost | USD 304.1929 |
| Matched task-blind expected quality | 0.9414687144 |
| Quality delta vs task-blind mean | +0.0016283799 |
| Matched task-blind expected cost | USD 306.4853 |
| Cost delta vs task-blind mean | -USD 2.2924 |
| Router quality percentile | 68.35 percent |
| Router cost percentile | 37.05 percent |

The external 0.99 point reaches the 96.71st quality percentile but costs USD 5.9150 more than the
matched task-blind mean. This is weak task-selection evidence, not a promotable result.

An Open-SWE-only native hashing follow-up is a negative result. It selected
`hash8192-ridge-heads-a10` with 0.0812 external out-of-fold uplift Spearman, but the external 0.97
point is 0.0012465 lower quality and USD 11.0395 more expensive than matched task-blind assignment
on DeepSWE. Its quality and cost percentiles are 33.26 and 94.12. Stronger source-domain
correlation therefore did not improve target transfer.

## Five-source structural transfer gate

The structural family was then refit across all five external sources with equal source weight.
This screen read no DeepSWE path. The selected `structural-irt-a100` candidate used 59.36 percent
strong traffic at the external 0.95 point, but its aggregate out-of-fold uplift Spearman was
-0.0167. Four sources had positive within-source uplift correlation and R2E Gym had -0.2766.

The preregistered cross-domain gate required positive aggregate uplift, better traffic than the
controls, and positive source-specific uplift on at least four of five sources. The negative
aggregate result rejects the family. No DeepSWE evaluation was run.

The frozen artifact has SHA-256
`e11b5fec3bfd72c7df1ba1496835b494f594b76275b5e113bd6a37e2a9c4ec52`.
Its selected research joblib is only 895 bytes and has SHA-256
`dedaff4f94efd55d6e1fd3eaff8784043744ab28732767870c1ca44a6d170e3c`.

## Latent task-profile follow-up

The next family adapts the task-type prior from TRouter without its inference-time language-model
classifier. SWE-rebench V2 provides external taxonomy supervision for difficulty, intent
completeness, and PR category. The profile teacher is a separate subset with no selected
Open-SWE paired outcome. Exact task identity and normalized prompt overlap are removed against all
five outcome sources before fitting.

A deterministic character-hashing classifier maps the request to soft profile centroids. A
source-balanced, shrunk table then estimates the marginal gain of the strong effort arm for each
profile. Serve time needs one local hash, centroid similarities, and a table lookup. It makes no
network call and persists no foundation model.

The family gate is frozen before its fit:

1. aggregate external out-of-fold uplift Spearman must be positive;
2. source-specific out-of-fold uplift must be positive on at least four of five sources;
3. leave-one-source-out uplift must be positive on at least four of five sources;
4. strong traffic at the external 0.95 point must be lower than both negative controls.

Only one mechanically selected candidate may open the DeepSWE artifact, and only if all four
conditions pass. A metadata-only audit found zero task-id and zero normalized-text overlap between
the 14,504 Open-SWE paired tasks and the 113 DeepSWE task metadata rows.

The completed fit selected `profile-difficulty-pr-t0.05-p20`, but rejected the family.
Aggregate out-of-fold uplift Spearman was -0.0731. Only two of five primary sources and
two of five leave-source-out checks were positive, below the preregistered four-source
minimum in both cases. No DeepSWE evaluation was opened for this family.

## Moonshiner fast execution corpus

The first execution-scored fast-development replacement used Moonshiner commit
`da981ff7893de29004712c8f4b3f7a414737525e`. A remote preflight required every selected
seed to fail before repair, pass after the reference repair, preserve protected files, avoid
network requirements, and have zero exact task-id or normalized-prompt overlap with DeepSWE.
It froze 96 tasks across 42 grouped families and eight language labels. The task artifact SHA-256
is `080910137aa29ab7ddee25920e9c314e523d916b94f1c2d8f88627952ab11819`.

All model calls used GPT-5.6 Luna with reasoning effort as the arm. The xhigh and max matrix
ran 192 cells for a trace-derived estimated USD 7.5222. The low, medium, and high matrix ran
288 cells for an estimated USD 6.8801. Including the valid smoke runs, measured trace-derived
spend was USD 14.5822, plus two ungradeable failed calls with no valid accounting.

One failed high-effort trace grew to 19.1 GB and filled the sandbox disk. Only that invalid raw
trace and its exact managed workspace were removed. The deletion is not recoverable, but the
failure log remains. The runner now enforces a 64 MB per-trace file limit and removes only its own
managed workspace after each cell.

This corpus was too easy and noisy for effort routing. Eighty-eight of 96 tasks passed at every
effort. For xhigh to max, 93 tasks tied, two improved at max, and one regressed. The selected
`latent-hash2048-a10` scorer had -0.0061 out-of-fold uplift Spearman, -0.00052 best advantage
versus matched task-blind routing, and bootstrap interval
[-0.01117, -0.00030, 0.01120]. The external gate failed and DeepSWE remained sealed.

## BeyondSWE trace-burden transfer

The next source used
[`AweAI-Team/BeyondSWE`](https://huggingface.co/datasets/AweAI-Team/BeyondSWE) task commit
`2dc9bab512c7dcb00397531da34e06572cf06674` and the repository's released GPT-5.4 XHigh
Codex trajectories at commit `9e25d5a15857c90fd1b63b674972879072fb78b5`. The task and
trajectory files each contain 500 rows. Two exception traces were excluded, leaving 498 rows.
No task-id or normalized-prompt overlap with DeepSWE or the Open-SWE validation source was
retained.

The source release has complete rewards, steps, and token counts but no measured `cost_usd`.
The freezer records that absence instead of imputing cost. It derives a dense trace-burden label
from graded reward deficit, log steps, log prompt tokens, and log completion tokens. No foundation
model or fitted model is persisted.

A structural ExtraTrees scorer predicted BeyondSWE burden with repository-grouped out-of-fold
Spearman 0.3756. Direct transfer to Open-SWE failed: uplift Spearman was -0.0071, the 20 percent
strong-traffic advantage was +0.00098, and its repository bootstrap interval
[-0.00188, 0.00096, 0.00352] crossed zero. DeepSWE remained sealed.

Source hashes:

| Artifact | SHA-256 |
| --- | --- |
| BeyondSWE tasks | `83a0e16ada8b6c95c5634c3ace41c75360b22f31735926f45273ae57623d021f` |
| Released Codex trajectories | `0211abac1057c218e71aee8d5cb162f472f4a4220fed22f2de359258950d9170` |
| Clean 498-row source | `fb4ca25b78c723066024dbc0bc5ec65bcbc795e49ae5b0ce5f2ff9d57e87dc30` |

## Zero-inflated Open-SWE uplift

Open-SWE has 2,805 positive-uplift tasks, 10,501 ties, and 1,198 negative-uplift tasks. The next
family therefore compared direct uplift, two-head, nonlinear structural, discordant-only ranking,
and a zero-inflated hurdle model. BeyondSWE burden was a fixed auxiliary feature selected before
Open-SWE labels were inspected.

Candidate selection was nested inside repository-held-out outer folds. The combined outer test
selected hybrid direct Ridge in three folds and the word sign ranker in two. It achieved:

| External metric | Result |
| --- | ---: |
| Uplift Spearman | 0.0942614 |
| Reward advantage at 20 percent strong traffic | +0.0084597 |
| Repository bootstrap 95 percent interval | [0.0051586, 0.0083905, 0.0116796] |

All external gates passed. Refit on the full external source selected
`hybrid-ridge-direct-a10`, with 0.0910 out-of-fold uplift Spearman and +0.00893 reward advantage.
The shuffled-label control reached only +0.00188.

The first DeepSWE replay accidentally used only title plus one-sentence display description,
while the external model was trained on the full runtime request. It is preserved as an invalid
feature-adapter replay. It scored 0.92660 at USD 196.80 and was worse than matched random quality.

The corrected replay built a label-free feature view from the full pre-inference task prompt,
froze all scores, and only then accessed reward and cost. At the frozen 20 percent max traffic:

| DeepSWE metric | Result |
| --- | ---: |
| Traffic | 88 Luna xhigh, 22 Luna max |
| Router reward | 0.9342591 |
| Router cost | USD 214.0205 |
| Quality delta versus matched random mean | +0.0027164 |
| Cost delta versus matched random mean | +USD 7.5256 |
| Quality percentile | 81.275 percent |
| Cost savings versus always max | 38.5734 percent |
| Quality retention versus always max | 98.7974 percent |
| Static dominators | 0 |

This is a functioning external Open-SWE selector, but it does not establish DeepSWE task-selection
value or a cost-efficiency win. The preregistered 95th quality-percentile and
matched-random-cost requirements failed.

## Trace-burden-aware cost ranking

An adaptive external-only follow-up subtracts a fitted multiple of BeyondSWE trace burden from
predicted Open-SWE uplift. The selection rule retains at least 90 percent of the best positive
inner-fold reward advantage, then minimizes predicted trace burden. It froze penalty 0.05.

The nested external result kept +0.0079886 reward advantage while reducing proxy burden by
0.0191567 per task. Repository bootstrap intervals were [0.0054980, 0.0104944] for reward
advantage and [-0.0221004, -0.0163907] for proxy burden. Every external quality and cost-proxy
gate passed.

The adaptive DeepSWE replay improved both axes relative to the unpenalized router:

| DeepSWE metric | Cost-aware result |
| --- | ---: |
| Router reward | 0.9345432 |
| Router cost | USD 211.5022 |
| Quality delta versus matched random mean | +0.0030005 |
| Cost delta versus matched random mean | +USD 5.0073 |
| Quality percentile | 83.565 percent |
| Cost savings versus always max | 39.2962 percent |
| Quality retention versus always max | 98.8274 percent |
| Static dominators | 0 |

It still fails promotion because target cost is above the matched random mean and target quality
is below the frozen 95th percentile. The penalty will not be tuned on DeepSWE.

## Final decision

Stop this lane. The strongest valid finding is a repository-held-out Open-SWE selection signal,
not established DeepSWE routing value. Valid trace-derived model spend was USD 14.5822, plus two
ungradeable failed calls without valid accounting. External fitting added no provider inference
spend.

The next experiment must compute a held-out oracle before building a router. A per-repository
expert pool is a plausible direction because repository identity is available before inference
and specialization may create real headroom. It may proceed only after its static-to-oracle gap
is clearly positive and materially large on held-out attempts.

Compact report hashes:

| Report | SHA-256 |
| --- | --- |
| Nested Open-SWE family gate | `b17e331d5d016bf9d1b9525e5b26a24f910d6aee7b04f77feb22913bfd1c583e` |
| Corrected full-prompt DeepSWE replay | `ff28df8a32718e8cf8132a5c52dc6123ecb7ef45692571c7fe7815a8c0b9b8d0` |
| Cost-aware external gate | `ac84740aed8918a5bb1e679a9567c9f9e97362acb58c04e6456f3b802942e1fd` |
| Cost-aware DeepSWE replay | `d1f810fe96b457e51f48c11d11127f59a9cf250169ef8224a02489d53a6f4d6a` |

## Reproducibility anchors

First external fit commit: `bbbaa609aa7f8b9e6a35aab311920ad11ef17266`

Frozen target grid report commit: `10cdd7c7`

Native policy commit: `3942ca6a`

Corrected source weighting commit: `20da7f5e`

Task-blind control commit: `962bf990`

`xhigh` model-pool contract commit: `1eef58e6`

First selected joblib SHA256:
`4eccda3b30b5f134691159cc003813e59d0a7dfe56e841742b3901988d599a96`

Frozen candidates SHA256:
`c8f3b96d62d82bf905b5446acf9b851485449f2283ecd3ec4305c61fba1f5fd8`

Frozen DeepSWE grid SHA256:
`2f6a22a2845bb0fec8f66233b7f26fd5b20f512950f6ef391f6bf216881a3cb2`

E2B template: `deepswe-router-cpu8gb-v1`

E2B sandbox: `idqwkvv60h7weldgl08p8`

Corrected native heads SHA256:
`88cbea68922457343781d2ba19c7404bf456e8b4686d0f1e8ea41beb170d4087`

WMO native policy SHA256:
`95826f5f31e3d2100208e734a03f96ff96fd8c9ea8a4517c424bde5cbc09e72f`

Serving parity report SHA256:
`b156a437e9ee70617c00d6ae62d4ce9511c91de9ce3c1993e619f82b69f34ba4`

Matched task-blind control report SHA256:
`7aaa55de211b5f1deb63c9437acf5440344a39a931fcb295b6018d5e285d86cc`

Original word-router task-blind control report SHA256:
`5dc64e22fbab0ebf10e8b62f5cbb81b90251ff4cc219b6992b2af119b402492c`

Open-SWE source preparation commit: `da8b1f8e`

Structural IRT family commit: `bd7f8f82`

Open-SWE compact source SHA256:
`179d9801507b514ec30eb279cf44235ee4b6634bf38b06cee741a1018c391d55`

Structural frozen candidates SHA256:
`4ca5cf101a358551fa077124a9f874854625ae9c00b5ef2e757546ac160f4efa`

Structural DeepSWE report SHA256:
`47ab1efc7e4556f6fd31b4af858073132a80fd569f5a65b2e11b31e094de3b4d`

Open-SWE native frozen candidates SHA256:
`919b60a85635629818ed67938abd594c5091839f8f64031669c2aa98f2800bee`

Open-SWE native DeepSWE report SHA256:
`ff4babf624c98f3fa6a5907a6d58f78666d73c674b7a7f7567508230794be46b`
