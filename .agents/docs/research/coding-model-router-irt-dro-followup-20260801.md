# Conditional graded IRT router follow-up

Status: conditional preregistration draft. This is not an active experiment and does not change
the frozen graded SWE-rebench v47 kNN protocol. DeepSWE outcomes remain sealed.

## Trigger and evidence boundary

This lane may start only if the current graded kNN study fails an external promotion gate. The
failure point determines the valid confirmation source:

1. If kNN fails on development before the current 320-task confirmation is opened, this study may
   use the completed development matrix for fitting and the still-sealed confirmation exactly
   once.
2. If kNN reaches and opens the current confirmation, those outcomes become development evidence
   after the kNN decision. A new repository-disjoint external confirmation cohort must then be
   frozen before this study is fit. The opened confirmation may not be reused as final evidence.

No DeepSWE outcome, target-derived threshold, or target rerun is permitted. Only the first
latency-neutral policy that passes a fresh external confirmation may receive one DeepSWE transfer.

## Hypothesis

Guarded kNN treats every nearby task as local evidence but does not explicitly represent the main
structure in this matrix: five ordered reasoning efforts for one model plus one frontier guard.
A graded item-response model may generalize better by jointly estimating task difficulty, task
discrimination, and arm ability from fail-to-pass counts. A distributionally robust decision rule
may then protect the quality constraint under repository shift without an inference-time model
call.

This combines two relevant ideas:

- IRT-Router models model ability and query attributes explicitly:
  `https://arxiv.org/abs/2506.01048`.
- RACER applies a KL-divergence uncertainty set to reasoning allocation under distribution shift:
  `https://arxiv.org/abs/2605.10805`.

POLLINATOR's graph plus IRT predictor is a secondary ablation, not the primary family:
`https://openreview.net/forum?id=N59cvpjnlo`. Online contextual-bandit routers are excluded because
the one-shot target transfer provides no legitimate target feedback.

## Frozen candidate family

The primary outcome is the exact binomial pair `(f2p_passed, f2p_total)`, not a binary resolve flag
or an unweighted reward fraction. For task `i` and arm `a`, fit

`f2p_passed[i,a] ~ Binomial(f2p_total[i], sigmoid(q[i] dot theta[a] - b[i]))`.

An exact task-level fit first estimates:

- scalar difficulty `b[i]`;
- nonnegative discrimination vector `q[i]`;
- one ability vector `theta[a]` per model by reasoning-effort arm.

Closed-form ridge then projects task difficulty and log discrimination onto pre-call features
inside each fit fold. Unseen tasks use only those projected feature heads and the ephemeral arm
abilities. The latent dimension is fixed at 2. Synthetic E2B preflight showed that dimensions 4
and 8 reached the nonconvex optimizer's iteration limit even before real outcomes, while dimension
2 completed. With six arms, eight independent ability dimensions are also unidentifiable.

The task encoder candidates are signed hashing at 512 and 2,048 dimensions, the frozen
prompt-shape vector, and a concatenation of the 2,048-dimensional hash with prompt shape. Signed
hashing uses stateless `HashingVectorizer` character-within-word n-grams of lengths 3 through 5,
signed collisions, and row L2 normalization. Prompt shape is the fixed ten-value view covering
log character, line, and word counts plus code-fence, exception, test, error, implementation,
refactor, and performance indicators. It accepts prompt text only. Projection regularization
strengths are 0.1, 1, 10, and 100. Compare unconstrained arm abilities with one variant whose Luna
capacity coordinate is monotone from low through max. The Sol guard remains a separate arm and
receives no ordinal constraint. The exact transform lives in
`.agents/scripts/coding_model_router_graded_irt_features.py`.

The graph ablation constructs a fit-only task similarity graph from the same pre-call features.
Feature rows are L2 normalized, each task selects its eight nearest tasks by descending cosine
similarity with stable task-index tie breaks, edge weights use shifted cosine `(1 + cosine) / 2`,
and the directed graph becomes an undirected union by taking the maximum edge weight. The resulting
combinatorial Laplacian propagates latent difficulty with penalties 0.01, 0.1, and 1 and changes no
online feature contract. Repository identity, model output, patch, tests, verifier details, and
future trajectory content remain forbidden features.

The graph and monotone variants are separate ablations, not a factorial cross. For each feature
view and regularization strength, the structural grid contains exactly five
variants: unconstrained without a graph, monotone Luna capacity without a graph, and unconstrained
with each of the three graph penalties. This produces 80 structural candidates before the 25
cost-penalty and KL-radius operating points.

## Implementation starting point

Reuse the tested optimizer and grouped-CV structure in
`.agents/scripts/coding_model_router_codeforces_irt.py`. Do not reuse its scientific assumptions
unchanged. The graded SWE-rebench adaptation must:

1. replace its equally weighted fractional cross-entropy with the exact binomial likelihood so a
   1-of-1 score does not carry the same information as a 100-of-100 score;
2. generalize the scalar difficulty and discrimination to the frozen two-dimensional latent;
3. replace direct linear or Chebyshev scalarization with the repository KL-robust selection rule;
4. remove fitted arm abilities and all other coefficients from persisted reports;
5. freeze only task identity, selected arm, input hashes, route provenance, and aggregate fit
   diagnostics in a label-free route manifest;
6. preserve its finite-difference gradient test, grouped split assertions, shuffled-label control,
   and latency audit.

The existing Codeforces implementation remains an historical experiment and is not modified by
this follow-up.

The prepared numeric core lives in
`.agents/scripts/coding_model_router_graded_irt_core.py`. It implements the exact count-weighted
binomial likelihood, analytic gradients, multidimensional nonnegative discrimination, a pre-call
feature-conditioned variant that can score unseen tasks, and the forward-KL repository robust
lower bound. Its promoted path fits exact task latents once and projects difficulty and log
discrimination onto pre-call features by exact dual ridge, including the graph penalty on
predicted difficulty. The earlier joint feature-conditioned optimizer remains tested numeric
infrastructure but is not in the promoted grid after its synthetic performance failure. The core
also implements the frozen monotone-capacity variant with a differentiable
cumulative-softplus parameterization on the first latent coordinate for Luna low through max; the
sixth Sol arm remains unconstrained. Its inline tests cover finite-difference gradients for both
ability variants, exact denominator weighting, unseen-task prediction, monotone Luna ordering, and
the KL solution. The feature-conditioned loss also accepts a validated fit-only graph Laplacian
and applies the frozen graph penalty directly to predicted task difficulty. Its tests cover the
graph gradient by finite differences, reject malformed Laplacians, and exercise the optimizer
wiring. The module has no filesystem or serialization surface. It remains conditional
infrastructure and does not activate this lane.

The pure protocol helpers in `.agents/scripts/coding_model_router_graded_irt_protocol.py` implement
seed-sensitive repository-disjoint folds with exact one-fold task coverage, the frozen cosine kNN
graph construction, and a shuffled-label control that permutes complete outcome rows only within
repositories. These replace the older Codeforces implementation's unseeded folds and corpus-wide
shuffle. They load no outcomes, fit no model, and persist no state, so preparing them does not
activate this lane.

The pure robust-selection helpers in
`.agents/scripts/coding_model_router_graded_irt_selection.py` implement the frozen pointwise guard,
repository reward and cost aggregation, forward-KL worst-case metrics, and paired robust margins.
They use only in-memory arrays and have no fitting, network, filesystem, or serialization surface.

The in-memory nested orchestrator in
`.agents/scripts/coding_model_router_graded_irt_nested.py` crosses the 80 structures with the 25
operating points inside every seeded repository fold. It fits both the real and within-repository
shuffled count rows, predicts each task out of fold exactly once, and retains only aggregate
metrics. It has no filesystem or serialization surface and does not activate this lane.
The same module exposes one-seed aggregate evaluation and a separate final selector so the five
frozen seeds can run in independent remote workers. The selector reconstructs the exact frozen
structure by operating-point grid and rejects missing, duplicated, or unexpected seed metrics
before it can promote a policy. No cross-fit probability, coefficient, or task embedding crosses
the worker boundary. A scientifically eligible candidate still cannot promote until its separate
single-core audit records at least 10,000 decisions, zero network calls, p50 below 5 ms, and p95
below 20 ms. Missing latency evidence is a failed promotion gate, not an implicit infinity used
only for tie breaking.

The remote runner in `.agents/scripts/coding_model_router_graded_irt_run.py` emits one complete
2,000-row scalar metric report per outer seed. It rejects incomplete grids, duplicate policies,
source drift, or mismatched audited inputs. A separate remote command refits only structures that
pass every non-latency gate, times the exact prompt feature, probability, and guarded-choice path
on one CPU, and emits aggregate latency rows only. Final selection requires exact latency coverage
of every scientifically eligible policy. The E2B orchestrator runs five seed workers concurrently,
then runs latency and selection in a no-internet sandbox. Every sandbox is terminated after its
bounded command, and only source hashes, input hashes, scalar metrics, latency aggregates, and the
selection decision return to the Mac.

Synthetic E2B performance preflight used 524 fit tasks, six arms, exact count likelihood, no
provider calls, no target outcomes, and no persisted fitted state. The original 512-feature,
two-latent joint fit failed after about 77 seconds at 1,000 iterations. Ridge initialization made
one joint case complete in 0.48 seconds, but representative joint graph and 2,048-feature fits took
38 to 64 seconds, and both four- and eight-latent task fits still failed. The projected
two-latent path completed task IRT in 0.24 seconds and every tested 512- and 2,048-feature
projection, including graph regularization, in 0.02 to 0.17 seconds. Aggregate timing artifact
hashes are `483b3eb8b146f466a7ba90bde5fb943fa1fc37ae058f5de245518dbdf7df733f`,
`673f86b3c83dbb25afbd6f62f48e5c499c0178e77eb168313155ad946b0bc4ac`,
`f5926272804df724ef0cd65865dcb83e4a83a8801a3c0599bebf9a168b04ecd4`, and
`320959137aa3749ec907b245187b90dea36ad772acf8c2b9a23dcce02e359ec1`.

The implemented projected core then received a second real-shape synthetic E2B preflight. Free,
monotone, graph-regularized, and 2,048-feature fits all converged in 0.35 to 0.70 seconds. The
complete 4,000-fit grid projects to 23 to 46 minutes serial and can be sharded independently by
the five frozen seeds. The preflight again used zero provider calls and no outcomes, persisted no
fitted state, and terminated the sandbox. Its aggregate artifact SHA-256 is
`d03357b5c29304c737b063ae0cc53642659b6d2f4d0de1024b22a54b0373c830` and its core source
SHA-256 is `0bdce68cfc85e872e7c424f41a37ffc87f6ed5411622217f5faac0263c231674`.

The first real development launch produced no seed artifact because one seed-23 fold reached the
L-BFGS-B 1,000-iteration limit. All five no-internet workers terminated and confirmation and
DeepSWE remained sealed. The optimizer now performs up to three deterministic continuation passes
from the previous finite iterate. The objective, initialization, bounds, tolerances, candidate
grid, folds, outcomes, and selection rule are unchanged. Non-finite results and every failure
other than the exact iteration-limit condition still fail closed. This is a numeric convergence
repair for the frozen search, not a new candidate or target-informed retry.

The repaired real development run completed all 4,000 frozen real and shuffled-label fits and
emitted 2,000 policy metrics for each of the five seeds. No policy passed development. Zero policy
in any seed reached 40 percent savings, and zero policy had a nonnegative robust cost margin. The
best nominal savings by seed ranged from 10.1 to 21.1 percent, and the best savings among policies
meeting the 95 percent quality floor ranged from 9.6 to 14.0 percent. Because there was no
scientifically eligible policy, the one-CPU latency worker correctly audited zero routes and no
confirmation manifest was produced. All six exact E2B sandbox identifiers returned
`SandboxNotFoundException` after teardown. The run made zero provider calls, persisted no fitted
state, and accessed neither confirmation nor DeepSWE outcomes. The selection, latency, and
manifest SHA-256 digests are
`4d1e29184c9b4db893a460cafc970382a782820b561faeb15102b111ca5e3067`,
`5394b16125b9c86a2693af2f090919033ec01b6354c31ff2231523a6ab7913f5`, and
`e3d77db37d4f271bbe59d43b85b1e12621347cfdec61dfbefbe75830fd4b6024`.

All fitting, cross-validation, bootstrapping, and latency measurement run on E2B or Azure. The Mac
only orchestrates and validates bounded artifacts. No foundation model, task embedding bank, or
fitted numeric router state is persisted. The remote worker may retain coefficients only for its
bounded fit process, freeze deterministic label-free route manifests and aggregate reports, then
destroy the worker and its coefficients.

## Robust selection rule

Use five repository-grouped outer seeds and five repository-grouped inner folds. Fit on inner
training repositories and predict every arm's graded pass probability on inner validation tasks.
For each candidate, enumerate cost penalties on the frozen grid `0, 0.005, 0.01, 0.02, 0.03`.

Within each inner fold, repeat the inner-training mean cost of each arm across validation tasks and
normalize those six costs to `[0, 1]`. For each task and cost penalty, choose the arm maximizing
`predicted_reward - cost_penalty * normalized_cost`, breaking ties by lower mean cost and then
frozen arm order. Revert that choice to the fit-selected static guard unless its predicted reward
is at least 95 percent of the guard prediction. This creates deterministic candidate policies
without weakening the pointwise predicted-quality guard.

Apply distributional robustness to each complete candidate policy, matching RACER's policy-level
formulation rather than claiming a per-task KL guarantee. Aggregate the paired task margins
`routed_reward - 0.95 * guard_reward` and `0.60 * guard_cost - routed_cost` within each repository.
The robust score is the minimum expected repository margin over the forward-KL set
`KL(shifted || equal_repository_empirical) <= radius`. Both lower bounds must be nonnegative.
Radius candidates are `0, 0.01, 0.03, 0.05, 0.1`. Radius and cost penalty are selected only from
inner out-of-fold routes and outcomes.

Radius zero is the required nominal ablation and cannot promote as the primary robust policy. A
promoted policy must pass at radius 0.01 or greater. When otherwise identical routes have the same
cost and pass at multiple radii, select the largest passing radius before applying the remaining
tie breaks. This prevents the mechanically weakest radius from winning every identical-route tie.

The mechanical winner minimizes cost subject to all of these fit-only conditions:

1. at least 95 percent quality retention in every outer seed;
2. at least 40 percent cost savings in every outer seed;
3. positive matched task-blind reward advantage;
4. positive within-repository shuffled-label advantage;
5. no static arm dominates it in both quality and cost;
6. no repository with at least five tasks loses more than 0.10 absolute reward;
7. route latency below 5 ms p50 and 20 ms p95 over 10,000 single-core decisions;
8. zero inference-time network calls.

The within-repository shuffled-label advantage is defined as mean actual routed reward from the
real-label out-of-fold policy minus mean actual routed reward from the otherwise identical policy
fit on complete count rows permuted within each repository. It must be strictly positive in every
outer seed. The matched task-blind control remains separate and uses the real policy's exact arm
traffic on every task.

Tie breaks are higher worst-seed retention, higher matched-blind advantage, lower latency, smaller
ephemeral coefficient count, lower latent dimension, and the frozen grid order.

## Required ablations

Report every static arm, task-blind effort mixing at identical arm traffic, cost-only routing,
seeded random routing, within-repository shuffled outcomes, guarded kNN, unconstrained graded IRT,
monotone-capacity IRT, graph-regularized IRT, robust radius zero, the selected robust radius, pair
oracles, and the full oracle.

An IRT result is not a routing gain unless it beats matched task-blind mixing with a positive
repository-bootstrap lower bound. Better calibration or latent interpretability alone cannot
promote it.

## External confirmation and target transfer

Refit the selected configuration ephemerally on external development, freeze its deterministic
label-free confirmation route manifest, then destroy the fitted state before opening confirmation
outcomes. Use 10,000 repository-cluster bootstrap draws with seed 20260801. Promotion requires at
least 95 percent quality retention, at least 40 percent savings, a nonnegative lower bound for
`router_reward - 0.95 * fit_selected_static_reward`, positive matched-blind advantage, no static
dominance, and route p95 below 20 ms.

Only after that report and every source hash pass may the policy produce a label-free DeepSWE route
manifest. Evaluate that manifest exactly once against the sealed graded DeepSWE matrix. Never tune,
repair, or rerun from target outcomes.
