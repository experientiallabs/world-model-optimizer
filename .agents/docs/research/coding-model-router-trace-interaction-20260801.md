# External trace-interaction representation for WMO kNN

Status: frozen before public trace download, representation fitting, or router fitting on
2026-08-01. DeepSWE outcomes remain sealed.

## Question

Can model-specific success differences learned from public coding-agent traces provide the task by
model interaction coordinate that generic difficulty, character task text, and off-the-shelf code
embeddings failed to recover?

The prior external experiments establish three facts. The 15-arm model by reasoning-effort matrix
contains large oracle complementarity. A generic trace-trained difficulty score predicts coding
success but not which arm supplies complementary success. A code-semantic WMO kNN policy preserves
quality only by routing roughly three quarters of tasks to `sol-max`, yielding about 10 percent
rather than 40 percent savings. This experiment learns a representation from cross-model residuals
in an independent trace corpus, then uses that representation only as the neighbor metric for
WMO's existing guarded router.

## Frozen public source

The only representation-training source is `chilomax/SWE-smith-trajectories`, config `default`,
split `tool`, converted-parquet commit `3bcb3dd101c2930e3386e0103dd9fde084587a1c`. A label-only
schema probe found 24,100 rows, 10,637 instance identities, and three source models:
`claude-3-5-sonnet-20241022`, `claude-3-7-sonnet-20250219`, and `gpt-4o-2024-08-06`.
There are 512 instance identities observed under multiple source models, including pair overlaps
of 471, 211, and 170 instances.

Only the canonical `<pr_description>` block from the initial user message may be used as text.
Assistant reasoning, tool output, patches, and post-task text are forbidden. Rows are aggregated
by exact `(instance_id, task-description-hash, source-model)`. A task variant enters pairwise
training only when that exact pre-call text was observed under both source models. Repository is
parsed from the instance identity and is the grouping key. If fewer than 100 exact prompt variants
exist for any source-model pair, that pair is omitted. If fewer than two pairs survive, the
experiment stops before touching the external model-effort development outcomes.

## Frozen interaction representation

Pre-call texts are embedded with `jinaai/jina-embeddings-v2-base-code` at immutable Hugging Face
commit `516f4baf13dec4ddddda8631e019b5737c8bc250`, using the 161,895,621-byte quantized ONNX
file `onnx/model_quantized.onnx` and same-revision tokenizer. Inputs are truncated to 1,024 tokens,
attention-mask mean pooled, and L2 normalized.

For each surviving source-model pair, the target is the difference between mean resolved rates on
the exact prompt variant. One additional target is mean resolution rate across all available
source models for the prompt. Ridge models map the 768-dimensional code embedding to these source
outcomes. The frozen alpha grid is 1, 10, or 100. Selection maximizes mean repository-grouped
Spearman correlation across five folds repeated at seeds 11, 23, 37, 41, and 59, then chooses the
smaller alpha. Source pretraining passes only if the aggregate mean correlation is positive, every
seed's mean is positive, and at least two pair-specific targets have positive aggregate
correlation.

The selected Ridge models are refit ephemerally on all eligible public trace variants. A new task's
interaction representation is the predicted generic resolution rate followed by the predicted
source-model pair differences in frozen lexical pair order. Coordinates are standardized by the
public-trace training mean and standard deviation, then the vector is L2 normalized. No target
model labels enter this representation.

## Frozen WMO development search

The interaction vectors replace the semantic vectors in the prior WMO experiment. All other
development choices remain fixed: the same 194-task, 15-arm SWE-rebench V2 matrix; the same static
baseline `sol-max`; the same eight guard arms; neighbor counts 8, 16, 32, or 64; asymmetric guard z
of 0, 0.5, 1, 1.645, or 2; `pick_lam` of 0, 0.01, 0.02, or 0.03; relative similarity threshold
0.95; no absolute novelty floor; minimum paired evidence 8; and the standard-error floor enabled.
This is the same frozen 640-candidate grid.

Evaluation uses repository-grouped five-fold cross-validation at seeds 11, 23, 37, 41, and 59.
A candidate is eligible only if every seed has at least 95 percent reward retention and at least
40 percent cost savings versus `sol-max`, mean matched task-blind reward advantage is positive, and
no static arm dominates it. Selection minimizes mean cost, then maximizes mean reward, then uses
frozen candidate order.

## Confirmation and target transfer

If development passes, refit the chosen source representation and WMO policy ephemerally, then
freeze one route for the untouched 200-task SWE-rebench V2 confirmation cohort, matched task-blind
traffic, and 128 repository-group null routes. The same external confirmation gates apply: at
least 95 percent whole-task coverage, 95 percent quality retention, 40 percent savings, positive
repository-bootstrap lower bounds versus both blind controls, no static dominance, and route
decision latency p95 below 5 ms.

Only a policy that passes every external gate receives exactly one DeepSWE v1.1 transfer. DeepSWE
may not be used for representation training, repair, hyperparameter selection, or a rerun.

## Compute, persistence, and spend

Public trace acquisition, ONNX inference, Ridge fitting, and WMO fitting run on ephemeral E2B or
Azure compute, never on the local Mac. No trace corpus, embedding model, task vector, Ridge weight,
kNN bank, or fitted numeric policy is persisted. Only aggregate reports and conditionally frozen
route decisions may return. No Modal app is used. This development stage makes no provider model
calls, so rough cumulative experiment spend begins at USD 3,025.10805955.

## Result

The source-only gate failed, so the model-effort development outcomes were not read. No provider
confirmation call was launched and DeepSWE remained sealed.

Exact canonical-prompt matching left 366 multi-model variants across only 10 repositories. The
Claude 3.5 versus Claude 3.7 pair had 208 variants, Claude 3.5 versus GPT-4o had 211, and Claude 3.7
versus GPT-4o had only 53, so the last pair was excluded by the frozen 100-variant floor. The two
eligible pair targets and generic source resolution target were evaluated at every frozen alpha.

All alphas produced negative mean repository-held-out correlation. Alpha 1 was the least negative
at mean Spearman -0.086865, with every seed below zero. Its Claude 3.5 versus 3.7 target was
-0.067087, Claude 3.5 versus GPT-4o was +0.013241, and generic resolution was -0.206748. Alphas 10
and 100 degraded further to -0.193853 and -0.247243. The public trace overlap is too sparse and too
repository-concentrated to learn a transferable cross-model task coordinate from semantic text.

The E2B sandbox `ip8tnc13eurfrtdv0z0ps` was terminated. Trace files, embedding model, vectors,
Ridge weights, and WMO banks were not persisted. Provider spend did not increase, so rough
cumulative experiment spend remains USD 3,025.10805955.
