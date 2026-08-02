# ASSERT-KTH external expert-pool oracle gate

Date: 2026-07-31

Status: final negative result

## Question

Does a public external coding-agent matrix contain enough repeatable task-by-arm interaction to
justify another latency-neutral router experiment?

This is an oracle screen, not a router fit. It uses no DeepSWE reward, cost, verifier output, or
trajectory. DeepSWE task metadata is used only to remove exact task-id and normalized-prompt
overlap.

## Source

- Dataset: `ASSERT-KTH/agentic-evals-artifacts`
- Dataset commit: `5db0c4b69382d160a313d7ceaded915398c63e13`
- Benchmark: SWE-bench Verified
- Task metadata: `princeton-nlp/SWE-bench_Verified`
- Task metadata commit: `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`
- Arms: two scaffolds, three models, and two temperatures
- Independent attempts: ten per arm

Only the approximately 50 KB official SWE-bench result JSON from each run may be downloaded for
this gate. Full trajectories and detailed predictions are excluded.

## Frozen arm roster

1. `nano-agent-Qwen_Qwen3-32B-temp0`
2. `nano-agent-Qwen_Qwen3-32B`
3. `nano-agent-agentica-org_DeepSWE-Preview`
4. `nano-agent-agentica-org_DeepSWE-Preview__temp0`
5. `nano-agent-mistral_devstral-2512`
6. `nano-agent-mistral_devstral-2512__temp0`
7. `r2e-gym-Qwen_Qwen3-32B`
8. `r2e-gym-Qwen_Qwen3-32B__temp0`
9. `r2e-gym-agentica-org__DeepSWE-preview`
10. `r2e-gym-agentica-org__DeepSWE-preview__temp0`
11. `r2e-gym-mistral_devstral-2512`
12. `r2e-gym-mistral_devstral-2512__temp0`

The `DeepSWE-Preview` string is a public model identity in the external source. It does not mean
that DeepSWE v1.1 target outcomes are used.

## Held-out oracle

For each of 400 deterministic attempt splits:

1. Use five attempts per arm to select the best arm for each task.
2. Use the other five attempts to score that choice.
3. Select the best static arm using only the five fit attempts.
4. Score the static arm on the same five held-out attempts.
5. Measure held-out oracle reward minus held-out static reward.

Tie breaking uses fit data only: global fit reward, then frozen arm order. The uncertainty
distribution combines attempt splits with repository bootstrap resampling. Repositories, not
individual tasks, are the sampling unit.

## Frozen gate

Proceed to cost extraction and router fitting only when every condition passes:

1. At least 250 tasks remain after exact task-id and normalized-prompt overlap removal.
2. At least eight arms have ten complete attempts on the same retained task cohort.
3. Mean held-out oracle headroom is at least 0.10 absolute reward.
4. The combined repository-and-attempt 95 percent interval lower bound exceeds 0.05.
5. Every retained arm has exactly one gradeable binary outcome per task and attempt. Missing,
   incomplete, empty-patch, and agent-error submissions score zero. Infrastructure omissions are
   not silently dropped.

The threshold is intentionally stronger than merely excluding zero. It seeks the ACRouter-shaped
regime that the saturated DeepSWE arm pool lacked.

## Stop rule

If the gate fails, do not inspect trajectory features, fit a router, extract per-task costs, or
open another DeepSWE replay. Preserve the report as a negative result and search for a different
external expert pool.

## Final result

The corrected remote run at source commit
`690d886a39745f4626c0a96f74234c0bcb661c33` evaluated:

- 500 retained tasks across 12 repositories
- 12 complete arms
- 10 attempts per arm
- 60,000 dense binary outcome cells
- 400 deterministic five-fit and five-heldout attempt splits
- 20 repository bootstraps per split

The label-free target feature view supplied 110 normalized DeepSWE prompts and 113 task ids.
Neither exact ids nor normalized prompts overlapped the 500 external tasks, so no external task was
removed. The run accessed no DeepSWE outcomes or costs and downloaded no full trajectories.

The held-out oracle headroom over the fit-selected static arm was `0.004937`, with a combined
repository-and-attempt 95 percent interval of `[-0.009794, 0.017622]`. The naive same-attempt
oracle headroom was `0.046`, leaving a `0.041063` winner's-curse gap.

The task-count, complete-arm, and dense-matrix gates passed. The mean-headroom and lower-bound
gates failed. The preregistered decision is therefore:

1. Do not extract costs.
2. Do not inspect trajectory features.
3. Do not fit a router on this pool.
4. Do not open another DeepSWE replay.
5. Search for a different repeated-attempt external expert pool.

No provider inference calls were made for this oracle screen. The final report SHA-256 is
`dc98502b0b89e2e94842a7860ab3604b5da2045c71844e6d1f197fd2ce6c4447`. The runner observed seven
active E2B sandboxes under the configured 1,000-sandbox cap and used the existing experiment-owned
sandbox.

## Failure audit

All intermediate failures were preserved and made zero provider inference calls:

1. Source commit `ce650b03` excluded valid official filenames containing `preds`.
2. Source commit `52a2fd1f` did not score nine explicit `incomplete_ids` as zero.
3. Source commit `321eb06a` indexed the full matrix during repository bootstrap resampling.
4. Source commit `a2df8d10` produced a complete report with zero normalized target prompts. That
   invalid report remains archived with SHA-256
   `5aea782b8da091895e281c9e08e17e506db2ad13f77373f8bb50fa3fb36bd949`.
