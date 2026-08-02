# Open-SWE trajectory-distillation protocol

Date: 2026-07-31

Status: frozen before the full trajectory scan

## Objective

Test whether post-run behavior from a large external coding-trace corpus can teach a small,
latency-neutral pre-inference scorer which tasks benefit from stronger reasoning effort. The
scorer receives only the task request at serve time. It may not inspect a model answer, tool
trace, verifier result, repository checkout, or DeepSWE outcome.

This is a new representation family. Earlier Open-SWE work fit direct lexical, structural,
profile, and zero-inflated uplift models from compact task outcomes. It did not distill the full
trajectory behavior into a task representation. The earlier target results remain audit history
and may not select a candidate, threshold, or feature in this lane.

## Frozen external sources

Trajectory source:

- dataset: `nvidia/Open-SWE-Traces`
- revision: `9c0e4579a4ee0effa3e5f7a552494a045f29377d`
- native files: 84 Parquet shards
- native Parquet bytes: 18,338,420,390
- published trajectories: 207,489
- scaffolds: OpenHands and SWE-agent
- model modes: MiniMax-M2.5 and Qwen3.5-122B

Canonical pre-call task text:

- dataset: `nebius/SWE-rebench-V2`
- revision: `475dd5e8703bb5fb22dd3c60b5d038b019eba1e0`
- native file: `data/train-00000-of-00001.parquet`
- native Parquet SHA-256: `0e0bf9355f892ad74ae98d4e1c404f39fd6654a8e351ee3e6ab162e4a64cd3ad`
- native Parquet bytes: 428,839,266

Source URLs must resolve through these immutable revisions. Dataset-server conversion refs may
be used only to discover the native shard layout, not as source identity.

## Remote-only trace summary

The 18.34 GB trajectory payload is streamed and reduced on E2B. It is never downloaded to the
local Mac and is not retained after the E2B worker is terminated. Process one pinned shard at a
time and persist only one compact row per trajectory plus a task-level aggregate.

The compact trajectory row may contain:

- task, repository, language, scaffold, model mode, trajectory id, and resolved flag;
- message, assistant-turn, reasoning-character, and content-character counts;
- tool-call count, distinct-tool count, repeated-call count, and maximum repeated-call run;
- shell, search, read, edit, and test-call counts derived from tool names and arguments;
- model-patch file and line counts;
- input shard identity and a digest of the source row.

It must not retain message text, reasoning text, tool arguments, tool output, reference patches,
or model patches. Reference-patch fields are excluded from every label and feature. Malformed
trajectories are counted and rejected explicitly rather than coerced to empty traces.

## Leakage boundary

DeepSWE rewards, costs, traces, verifier output, embeddings, and arm outcomes remain unread. A
label-free DeepSWE view may be used only to remove exact task-id and normalized-request overlap.
The report must record the overlap counts and prove `target_outcomes_used: false`.

Before fitting, a completeness audit found that the earlier full-prompt target
view covered 110 of 113 tasks. The label-free seal was therefore rebuilt from
those 110 requests plus the three missing public `instruction.md` files in the
pinned DeepSWE task archive. The resulting 113-row view has SHA-256
`35ad33855f63f147b1861b58b59ad635f8860677b5d0d5e902c421029d78637b`.
Every row contains exactly task id, repository, and request text. Its explicit
cost- and reward-access flags are false. This amendment happened during the
trace-only source scan, before any trajectory fit or new target evaluation.

Repository identity groups every split. A repository may not cross train and validation within
an outer or inner fold. Duplicate task ids and normalized requests stay in one group and receive
one task weight regardless of trajectory count.

## Frozen algorithm family

For each task, aggregate trace summaries separately by scaffold and model mode, then form an
agent-neutral burden signature from robustly scaled medians and interquartile ranges. Resolution
is not part of the burden signature. The signature contains no reference-patch information.

Compare three fixed pre-inference families:

1. direct signed character-hash two-head Ridge, the existing baseline;
2. trajectory-distilled burden features only;
3. direct hash features concatenated with out-of-fold predicted burden features.

The burden predictor maps task text to the multi-output signature with Ridge regression. Hash
dimensions are 2,048 and 8,192. Burden and reward-head alphas are 1 and 10. Every predicted
burden feature used for a training row must be generated out of fold. Outer-heldout predictions
come from a predictor fit only on outer-training repositories.

Candidate and routing-threshold selection is nested inside repository-held-out outer folds.
Thresholds target 0.95, 0.97, and 0.99 quality retention relative to the stronger external arm.
The selected point minimizes strong-arm traffic among feasible points, then maximizes reward,
then uses the frozen family and grid order.

Negative controls are matched task-blind traffic, repository-wise shuffled arm outcomes, direct
hash without distilled features, weak static, and strong static. Unattainable per-task oracle
headroom is reported with held-out attempts when the source provides multiple attempts.

## External gate

The trajectory-distilled family advances only when all conditions hold on pooled outer-heldout
predictions:

1. reward advantage over matched task-blind routing has a repository-bootstrap 95 percent lower
   bound above zero;
2. uplift Spearman is positive;
3. quality retention is at least 0.95 in every outer fold;
4. strong traffic is at least 20 percent below strong static;
5. the shuffled control fails the same gate;
6. the selected distilled family is not worse than the direct-hash baseline on both reward and
   strong traffic;
7. every source row is gradeable and no target outcome was read.

If the gate fails, preserve the negative result and do not evaluate DeepSWE. A passing result
freezes exactly one route rule for an independent external confirmation before any new DeepSWE
transfer. No target refit, threshold adjustment, or representation change is permitted.

## Compute and artifact policy

All trace scans, bootstraps, and fits run on E2B or Azure. The Mac is limited to editing,
orchestration, compact report sync, and lightweight tests. No Modal app is used. No foundation
model is downloaded or persisted. Fitted numeric state remains ephemeral unless a candidate
passes every external gate; even then, only the minimal serving coefficients may be retained.

The user authorized a USD 20,000 hard ceiling and monitors provider usage externally. This source
scan and fit require no provider inference spend. Existing paid matrices continue with exact
resume semantics and never rerun completed cells.
