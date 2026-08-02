# External trace-trained difficulty router

Status: frozen before trace download or fitting on 2026-08-01. DeepSWE outcomes remain sealed.

## Question

Can a generic coding-task difficulty score learned only from public SWE-smith agent traces recover
the model by reasoning-effort oracle headroom that direct fitting on 194 SWE-rebench task texts
could not capture?

This is a distinct sequential hypothesis. The prior model by effort experiment established that
pairwise oracle headroom exists but that small in-domain task-text Ridge and kNN estimators do not
generalize. This experiment tests whether thousands of independently generated coding-agent
trajectories provide a better task prior. It does not use DeepSWE rewards, costs, gold patches, or
tests.

## Frozen trace source

The only trace-training source is `chilomax/SWE-smith-trajectories`, config `default`, split
`tool`, at converted-parquet commit `3bcb3dd101c2930e3386e0103dd9fde084587a1c`. The frozen split
contains 24,100 rows in eight Parquet shards. Dataset Viewer reports the columns `messages`,
`instance_id`, `resolved`, `model`, `traj_id`, and `patch`.

Only the `<pr_description>` task block in the initial user message is extracted from `messages`.
Wrapper instructions may vary across trajectories and are excluded. Assistant reasoning, tool
responses, patches, and any post-task text are forbidden as model inputs. The source contains
genuine task-description variants under some shared instance identities. Rows are therefore
grouped by exact `(instance_id, task-description-hash)`; duplicate exact task prompts become one
training example whose target is mean `resolved`. Repository is parsed from the instance identity
and is the grouping key, so all variants from a repository stay in the same fold. Acquisition and
all fitting run on ephemeral E2B storage. No trace corpus or fitted weights are written to the Mac.

## Frozen external pretraining

The trace-only estimator is Ridge over signed character 3-to-5-gram hashing. The frozen grid is
2,048, 8,192, or 32,768 dimensions and alpha 1, 10, or 100. Selection uses five repeated
repository-grouped five-fold evaluations with seeds 11, 23, 37, 41, and 59. The metric is mean
Spearman correlation between predicted and observed per-instance resolution rate. Ties break by
smaller dimension and then smaller alpha. If mean correlation is not positive and every seed is
not above zero, the experiment stops before using the model-effort development labels.

The selected configuration is then refit ephemerally on every frozen trace instance and produces
one scalar ease score for each SWE-rebench development and confirmation task. No fitted numeric
state is persisted. Only aggregate reports and final route decisions may leave E2B.

## Frozen routing search

The quality guard is the already established external static arm `sol-max`. Each of the other 14
arms is considered as the alternate. A candidate sends the easiest tasks to the alternate and all
other tasks to `sol-max`. Frozen thresholds are score ranks 5, 10, 15, through 95 percent on the
194-task development cohort. This yields 266 candidates. Candidate selection minimizes mean cost,
then maximizes mean reward, then uses frozen alternate-arm and threshold order.

A candidate is eligible only if the full development cohort has at least 95 percent reward
retention versus `sol-max`, at least 40 percent cost savings, positive matched task-blind reward
advantage, and no domination by any static arm. If none is eligible, the result is terminal
negative and DeepSWE remains sealed.

## Blind external confirmation

If development passes, freeze exactly one route for the untouched 200-task SWE-rebench external
confirmation cohort, plus a task-blind route with identical arm traffic and 128 unique null routes
formed by repository-group permutation of the confirmation ease scores. Run only the selected two
model by effort arms, with two attempts per task under the pinned official verifier.

Promotion requires all of:

1. at least 95 percent whole-task coverage with whole-task intersection only;
2. at least 95 percent reward retention and at least 40 percent cost savings;
3. positive repository-bootstrap lower bound versus the matched task-blind route;
4. positive repository-bootstrap lower bound versus the best of 128 frozen null routes;
5. no domination by an eligible static arm;
6. route lookup latency p95 below 5 ms;
7. zero DeepSWE outcome access and no persisted trace model or fitted numeric router state.

If every gate passes, the frozen route receives exactly one DeepSWE v1.1 transfer. No target-based
repair or rerun is allowed.

## Spend and compute

The user authorized a USD 20,000 hard ceiling and monitors provider usage externally. Trace
acquisition, feature construction, fitting, and analysis run on E2B or Azure. The local Mac only
orchestrates and validates small artifacts. No Modal app is used. Rough cumulative experiment
spend before this hypothesis is USD 3,025.10805955.

## Result

The result is terminal negative on external development. No provider confirmation call was
launched and DeepSWE remained sealed.

The immutable trace split contained 24,100 unique trajectories, 10,637 instance identities,
11,937 exact instance-prompt variants, and 129 repositories. The selected trace-only estimator was
32,768-dimensional hashing with alpha 10. Its mean repository-grouped Spearman correlation was
0.288334, with all five seeds between 0.281599 and 0.292890. The public trace data therefore did
contain reproducible generic task-difficulty signal.

None of the 266 frozen difficulty-rank routes passed development. The highest-quality near miss
sent the easiest 5 percent of tasks to `luna-xhigh` and the rest to `sol-max`. It retained 98.59
percent reward, but saved only 6.37 percent cost and gained only 0.002843 reward versus matched
task-blind traffic. More aggressive routes lost quality before approaching the required 40 percent
savings. The trace difficulty score predicts generic SWE-smith success but does not identify which
model by effort arm supplies complementary success on SWE-rebench.

All Parquet data and fitted weights remained ephemeral on E2B. Only the aggregate negative report
was returned. Provider spend did not increase, so rough cumulative experiment spend remains USD
3,025.10805955.
