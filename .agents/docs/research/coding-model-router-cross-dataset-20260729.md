# Cross-dataset reasoning-effort router

Status: strict zero-shot transfer completed; two post-hoc external-transfer formulations completed
as exploratory analysis.

Date: 2026-07-29

## Question

Can a router learn task hardness from public coding-agent traces outside DeepSWE, then select
reasoning effort on DeepSWE v1.1 without training on DeepSWE outcomes?

The action space is reasoning effort within a fixed model family. Model-family results are
secondary sensitivity checks.

## External training data

The selected source is `nebius/SWE-agent-trajectories` at revision
`68195a1450865274106246d0d0296a1d6807b88e`, licensed CC BY 4.0. It contains 80,036 verified
trajectories. The experiment uses the 527 public tasks that have:

1. repeated `swe-agent-llama-8b` outcomes;
2. repeated `swe-agent-llama-70b` outcomes;
3. a public issue definition in `nebius/SWE-bench-extra` or the SWE-bench development split.

One paired task, `RedHatInsights__insights-core-3949`, is excluded because neither public task
corpus contains its issue definition. The exclusion is persisted before fitting.

The paired external means are 0.07445 for 8B and 0.11155 for 70B. The five-level hardness target
is the 70B verifier reward. This target had the strongest repository-grouped out-of-fold
predictability among the inspected raw success labels. The selected weighted kNN uses `k=20` and
has mean repository-grouped Spearman 0.0951 across five seeds.

A second external-only fit predicts the per-task 70B minus 8B reward advantage. It selects `k=40`
using five repository-grouped seeds, then chooses the least 70B traffic that retains at least 95
percent of 70B quality in seed 37 out-of-fold predictions. The frozen threshold retains 95.28
percent with 77.61 percent strong-model traffic. The transfer hypothesis maps predicted scaling
benefit to high rather than low reasoning effort.

An effort-labeled alternative, `greghavens/glm-5.2-coding-and-debugging-traces`, was inspected but
rejected as counterfactual router supervision. Its 207 trajectories contain one effort per task,
166 use max effort, and there are no outcomes for alternate efforts on the same task.

## Leakage controls

- External labels, model choice, neighborhood size, absolute novelty floor, and effort traffic
  masses are fitted before DeepSWE outcomes are opened.
- Router features contain only task text available before inference.
- The 110 DeepSWE embeddings are reused from an independent cache. No target embedding API call
  is made.
- DeepSWE rewards and costs are used only for evaluation.
- The strict result is confirmatory.
- The rank-normalized result is exploratory because it was designed after the strict run exposed
  a feature-distribution shift. Its calibration uses target embeddings only, never target rewards
  or costs.
- The binary scale-benefit result is also exploratory. Its threshold is fitted entirely on
  external outcomes and each target decision is independent, but the formulation followed the
  strict target result.

## Frozen effort policy

The predicted ease bins assign the easiest 30 percent to low effort, then 25 percent to medium,
20 percent to high, 15 percent to xhigh, and the hardest 10 percent to max. If a family lacks a
requested level, selection collapses upward to the next available effort.

The strict policy also sends tasks below the external tenth-percentile nearest-neighbor similarity
to the highest available effort.

## DeepSWE v1.1 results

The primary family is `mini_swe_agent_claude_opus_5`. Its best static effort on the published
110-task matrix is high, with graded reward 0.95433 and cost USD 679.86.

| Policy | Confirmatory | Graded | Quality retained | Cost | Savings | Effort mix low/medium/high/xhigh/max |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Best static high | reference | 0.95433 | 1.0000 | USD 679.86 | 0.00% | 0/0/110/0/0 |
| Strict absolute transfer | yes | 0.94439 | 0.98959 | USD 1,179.94 | -73.55% | 3/7/6/10/84 |
| Rank-normalized repair | no | 0.94154 | 0.98660 | USD 641.62 | 5.63% | 28/25/21/16/20 |
| Binary scale-benefit transfer | no | 0.94802 | 0.99339 | USD 427.20 | 37.16% | 53/0/57/0/0 |

For the strict policy, 75 of 110 tasks fall below the external absolute similarity floor. The
guard therefore routes most tasks to max effort. It preserves quality but fails the cost objective.

The exploratory repair recalibrates the frozen traffic masses on the unlabeled DeepSWE feature
distribution and uses the target-relative tenth similarity percentile as its novelty floor. It
reduces abstentions to 11 and saves 5.63 percent. Its paired repository bootstrap interval for the
cost ratio is 0.938x to 1.221x, so break-even is not established. Its graded delta is -0.01278
with a 95 percent interval of -0.03086 to 0.00402.

The binary scale-benefit router uses no target-wide calibration. Each task independently routes to
low or high using the external advantage threshold. It saves 37.16 percent and retains 99.34
percent of static high quality. Its paired repository bootstrap interval for the cost ratio is
1.409x to 1.806x, so savings are established. Its graded delta is -0.00630 with a 95 percent
interval of -0.02594 to 0.01028.

## Conclusion

A cross-dataset reasoning-effort router is working. The best exploratory policy sends 53 DeepSWE
tasks to low effort and 57 to high effort, retains 99.34 percent of the strongest static quality,
and saves 37.16 percent with a cost-ratio interval entirely above break-even.

It is not promotion-ready. It misses the original 40 percent point-estimate savings gate by 2.84
points, and its formulation was selected after observing the strict target result. The strict
confirmatory policy preserves quality but fails cost because its absolute OOD guard does not
transfer across corpora.

The next high-value step is to train on an external corpus with repeated verified outcomes for
multiple reasoning efforts on the same tasks. Accepted traces with one chosen effort do not
identify the counterfactual effort benefit. The binary policy should be frozen and tested on a
fresh DeepSWE task split or another untouched coding benchmark before promotion.

## Reproduction

Runner:
`.agents/scripts/coding_model_router_cross_dataset.py`

Tests:
`.agents/scripts/coding_model_router_cross_dataset_test.py`

Ignored durable artifact root:
`.wmo/experiments/coding-router-cross-dataset-20260729`

The run made nine external embedding requests covering 527 tasks and 142,170 input tokens. It
made no model episodes and no target embedding requests. At the published
`text-embedding-3-large` rate of USD 0.13 per million tokens, the rough incremental cost is
USD 0.0185.
