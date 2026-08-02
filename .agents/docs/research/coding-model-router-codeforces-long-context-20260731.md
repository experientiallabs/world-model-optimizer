# Codeforces long-context effort development protocol

Status: complete negative external confirmation on 2026-07-31; DeepSWE remains sealed.

## Objective

Correct the execution confound discovered in the first Codeforces source run,
develop a latency-neutral effort router on the original 160-task development
cohort, and evaluate the frozen route once on a separate untouched Codeforces
confirmation cohort before any DeepSWE transfer.

## Development corpus and boundary

The development corpus remains the 160-task `open-r1/codeforces-cots`
`solutions_py_decontaminated` cohort at revision
`39ac85c150806230473c70ad72c31f6232fe3f41`, with task SHA-256
`c99ac2b6637cc3c689f0c105938bc2932a40d7b3e9ed738239e10fa2b3c764c6`.
Published generations remain unloaded. DeepSWE outcomes remain sealed.

The first source run showed that 32,768 output tokens truncated 85 max and 33
xhigh cells. This development protocol raises `max_output_tokens` to 131,072
for every effort. All other prompts, tests, model identity, isolation, reward,
cost provenance, and retry rules remain unchanged.

## Corrected development matrix

- Model: `gpt-5.6-luna`
- Efforts: `low`, `medium`, `high`, `xhigh`, `max`
- Attempts: two
- Tasks: 160
- Cells: 1,600
- Maximum output tokens: 131,072
- Reward: fraction of frozen tests passed

Run one four-cell smoke first on two development tasks whose earlier high
effort calls were truncated, at xhigh and max, attempt 0. The smoke must prove
completed provider status, exact model attestation, gradeable outcomes, raw and
code hashes, and zero-pending resume. If the corrected smoke still truncates,
stop without launching the matrix.

## Development and confirmation

The corrected development outcomes may be used adaptively to choose features,
algorithm family, thresholds, and arm set. They may not authorize DeepSWE.
After development, freeze exactly one lightweight route rule and a separate
160-task confirmation cohort from previously unused eligible Codeforces tasks.
The confirmation cohort must exclude every development task and remain unseen
until the rule and promotion criteria are frozen.

Promotion requires positive reward advantage over the matched task-blind
mixture, a positive contest-cluster bootstrap lower bound, no static-arm
dominance, a failed shuffled-label control, complete grading, and no target
leakage. Only a passing untouched confirmation can authorize the single
DeepSWE transfer.

## RADAR-inspired development candidate

During the corrected development matrix, a primary-literature review identified
RADAR (`arXiv:2509.25426`) as a direct match for the observed model x reasoning
effort structure. This is an adaptive development candidate, not a new holdout
claim. It fits a regularized two-parameter item-response model to the graded
test-pass fractions, predicts task difficulty and discrimination from signed
character hashes plus the frozen structural features, and routes with the
paper's linear or Chebyshev performance-cost scalarization.

The nested contest-grouped grid is fixed at hash dimensions 512 and 2,048,
Ridge alphas 1 and 10, scalarization weights 0.70, 0.80, 0.90, 0.95, 0.98, and
0.99, and both scalarization families. Selection minimizes cost among points
within 0.5 reward points of the fold-selected static arm and requires positive
matched task-blind advantage. The development report cannot authorize
DeepSWE. If its grouped gate passes, the consensus rule is frozen and evaluated
once on the already required untouched 160-task Codeforces confirmation cohort.
No fitted numeric model is persisted.

## Corrected development evidence

The corrected matrix completed all 1,600 cells. Its combined outcomes have
SHA-256 `da1491fa7c1da93124e9ab14ac14c026e257d8baf85815e4499b4ec27446e2a9`.
The five static effort arms measured as follows:

| Effort | Mean reward | Total cost, USD |
| --- | ---: | ---: |
| low | 0.80854 | 1.5436 |
| medium | 0.87030 | 2.6149 |
| high | 0.92390 | 6.8878 |
| xhigh | 0.95406 | 11.8876 |
| max | 0.97813 | 21.1883 |

Three development families were evaluated with contest-grouped outer folds:

- Native kNN failed because its shuffled-label control also had a positive
  contest-bootstrap lower bound.
- The RADAR-inspired IRT route improved over its matched task-blind mixture,
  but failed its frozen uplift-correlation condition.
- The direct sparse effort route passed all development conditions. Its
  outer-heldout reward was 0.97016 at USD 15.7504, compared with 0.95210 for
  the matched task-blind arm mixture. The advantage was 0.01806 with a
  contest-bootstrap 95 percent interval of `[0.00749, 0.01801, 0.02908]`.
  Its shuffled control failed, and no static arm dominated it.

The corrected direct report has SHA-256
`ba57bc38ab0dafe500654014e22627c6916347513022729cbde738cfb0dd07dc`.
It authorizes confirmation only and explicitly leaves DeepSWE unauthorized.
The full task-text hashing, structural scaling, four Ridge heads, and arm
selection path took 462.45 ms per 160-task batch on the E2B reference worker,
or about 2.89 ms per task. No fitted model was persisted.

## Frozen confirmation rule and cohort

The exact confirmation route is `direct-hash512-a10-t0`: signed character
3-to-5-gram hashing with 512 features, the frozen structural features, four
Ridge delta heads with alpha 10 relative to `luna-high`, threshold zero, and
the cheapest-arm tie break. It is fit only on the complete corrected
development matrix. No confirmation outcome may change its representation,
regularization, threshold, arm roster, or tie break.

The untouched confirmation cohort was then selected from the same pinned
`open-r1/codeforces-cots` revision. It contains 160 previously unused tasks,
140 contest groups, and exactly 40 tasks in each of C, D, E, and F+.
Its task artifact has SHA-256
`16746ede6cd2b853da0d11889e4edbb1d08262ffe7aebb98aa344215f045cf59`;
its manifest has SHA-256
`b187986bd517b003ec8c7769ead9922e5d3c6b4a1ae8d45fee0cdf97597d8dfa`.
The manifest proves zero development-task overlap, zero normalized DeepSWE
prompt overlap, no published-generation access, and no target-outcome access.

Run the same five effort arms, two attempts, verifier, and 131,072 output-token
limit for exactly 1,600 confirmation cells. The frozen route advances only if:

1. every cell is gradeable and target sealed;
2. route reward minus the matched task-blind mixture with the same arm counts
   is positive and its contest-bootstrap 95 percent lower bound is above zero;
3. no single static effort has at least the route reward at no greater cost;
4. the same route fit after a fixed development-label permutation fails the
   primary matched-blind advantage gate; and
5. the full pre-inference path remains below 5 ms per task on the reference
   E2B worker.

Only a passing untouched confirmation authorizes the one sealed DeepSWE
transfer. A failure is final for this route and cannot be repaired with
confirmation-dependent tuning.

## Confirmation result

The untouched matrix completed all 1,600 cells. Its combined outcomes have
SHA-256 `fb2fec2c2e23a1c44867648fa622c0827bb1ea01c4b9c518df0abb8cf403b1e8`.
Seven provider responses ended at the frozen 131,072 output-token limit and
were retained as gradeable zero-reward model outcomes. The five effort arms
measured as follows:

| Effort | Mean reward | Total cost, USD |
| --- | ---: | ---: |
| low | 0.78719 | 3.0123 |
| medium | 0.86141 | 5.9876 |
| high | 0.91875 | 16.1243 |
| xhigh | 0.95547 | 27.7140 |
| max | 0.97125 | 48.4895 |

The frozen `direct-hash512-a10-t0` route achieved reward 0.96078 at USD
22.0005. Its matched task-blind mixture with identical arm counts achieved
reward 0.95687 at USD 19.4351. The positive point advantage of 0.00392 did not
survive contest-grouped resampling: the 95 percent interval was
`[-0.01010, 0.00387, 0.01650]`. The route was not statically dominated, its
shuffled-label control failed as required, and its full pre-inference path took
466.45 ms per 160-task batch, or about 2.92 ms per task.

The confirmation report has SHA-256
`24da2239fdd6a279fdd20a018dec9aaa0108eebbf1b0e08df8fbc6282d8341c4`.
The only failed gate was the required positive bootstrap lower bound. This
failure is final for the frozen route. No DeepSWE route was frozen and no
DeepSWE outcome was opened.

## Preregistered single DeepSWE transfer

This section records the preregistered transfer that was not executed. If and
only if every confirmation condition had passed, refit the exact frozen
candidate on the 160 development tasks and score the 113-row label-free
DeepSWE task view with SHA-256
`35ad33855f63f147b1861b58b59ad635f8860677b5d0d5e902c421029d78637b`.
Write and content-address all 113 effort decisions before opening any target
outcome. The route-freeze phase cannot accept a target matrix and runs on a
no-internet E2B worker. Its fitted Ridge heads remain ephemeral.

Codeforces-only tests, time limits, memory limits, and C/D/E/F+ bucket fields
do not exist on DeepSWE. The target adapter is therefore frozen before target
evaluation: preserve the shared text length, line, formatting, keyword, and
topic features, and set those seven unavailable source-only fields to zero.
Neither the adapter nor any effort decision may change after target outcomes
are opened.

Evaluate the frozen decisions exactly once against the already content-addressed
published DeepSWE matrix with SHA-256
`2988742e48b1c9bfec8dc45d88af112c46c45367529d1936b709e4b4e549835f`.
Use graded fail-to-pass reward and measured trial cost. Drop a target task
across all five effort arms if any reward or cost cell is missing. Report every
static effort, the router, and the matched task-blind mixture with identical
effort counts. Bootstrap the router-minus-blind reward by repository.

The transfer counts as positive task-routing evidence only if the repository
bootstrap lower bound is above zero and no single static effort has at least
the router reward for no greater cost. Regardless of the result, there is no
second target evaluation and no target-dependent refit, threshold change,
representation change, or arm change.
