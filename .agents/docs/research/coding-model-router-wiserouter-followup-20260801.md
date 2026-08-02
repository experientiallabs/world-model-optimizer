# Workload-budget graded router follow-up

Status: frozen after the graded kNN and IRT development failures, before implementation or fitting.
The 320-task external confirmation and every DeepSWE outcome remain sealed.

## Question

Can workload-level budget allocation recover the external pair-oracle headroom that per-task WMO
guards and projected IRT left unused, without an inference-time model call or added model latency?

The 649-task graded SWE-rebench development matrix contains a sharp gap. Every pair oracle that
includes `sol-max` saves 56.9 to 61.4 percent while improving reward, but the quality-safe kNN
region saves only 11.5 to 13.4 percent and projected IRT saves at most 14.0 percent at the quality
floor. Both failed because they reverted too many individual tasks to the expensive guard.

WISERouter formulates routing as a constrained contextual bandit and allocates a shared workload
budget with adaptive linear programming instead of imposing an independent cost decision on each
query: `https://arxiv.org/abs/2607.23765`. Its reported SWE-Bench result motivates this distinct
family, but this study uses the stricter repository-grouped, graded, matched-blind, null-control,
and external-confirmation gates already frozen for this experiment. CR2's marginal risk-control
framing is a secondary motivation for the explicit false-cheap null gate:
`https://arxiv.org/abs/2605.12001`.

## Evidence boundary

Development reuses only the audited 649-task, six-arm, one-attempt graded matrix with outcomes
SHA-256 `5023abf4e16d52a0a324bead44b1db80b518c2408593189585f0a4b416f94822` and audit
SHA-256 `d256c23e6661d9b1ef232c74d52922b5c1ce83a69ca1c87e723c97266a71064b`.
The arms remain `luna-low`, `luna-medium`, `luna-high`, `luna-xhigh`, `luna-max`, and `sol-max`.
The confirmation manifest remains the unopened 320-task repository-disjoint cohort with SHA-256
`c9443c9956e496123f396ee793efbb3368312092c4dcbd4e5e10bb77bd814f0a`.

No confirmation reward, DeepSWE reward, target cost, patch, test, verifier output, model response,
or trajectory may enter development fitting or routing. The only task inputs are repository,
language, and the initial issue text available before the first provider call.

## Frozen context family

Construct a stateless signed character 3-to-5-gram hash of the exact pre-call task text. Hash
dimensions are 512 and 2,048. Append the existing fixed 15-value SWE-smith prompt-shape block:
log character, word, line, code-fence, stack-marker, path-token, quoted-token, fix-term, test-term,
and dependency-term counts plus Python, JavaScript, TypeScript, Rust, and Go indicators.
Standardize only this dense shape block on fit rows, concatenate it to the L2-normalized hash,
then L2-normalize the combined vector.

Fit spherical K-means on fit repositories only, with 8, 16, or 32 contexts. Initialization is
`k-means++`, `n_init=20`, maximum 300 iterations, and the outer split seed. A candidate is invalid
for a split if any fit context contains fewer than eight tasks. Held-out tasks are assigned to the
nearest fit centroid by cosine distance. No embedding API, local foundation encoder, or network
call is permitted.

For each context and arm, estimate mean reward and mean cost from fit tasks. Shrink both estimates
toward the fit-wide arm mean with 0, 4, or 16 pseudo-observations. These are the only estimator
variants.

## Frozen workload allocator

For each fit context distribution, solve the WISERouter offline linear program that maximizes
expected graded reward subject to expected cost. The five workload budgets target 40, 45, 50, 55,
and 60 percent savings against the fit-selected strongest static arm. The null action is forbidden
because every task must be routed. Each context's arm probabilities sum to exactly one.

Turn fractional probabilities into deterministic routes without labels. Within each context,
order held-out task identities by SHA-256 of
`wiserouter-v1:<outer-seed>:<fold>:<context>:<task-id>`. Use systematic rounding with a
deterministic offset derived from the same prefix so each context-arm count differs from its
fractional target by at most one task. Ties use lower expected cost, higher expected reward, then
frozen arm order. No held-out reward or realized cost enters rounding.

The grid has exactly 90 candidates: two hash dimensions, three context counts, three shrinkage
values, and five workload budgets.

## Development selection

Use repository-grouped five-fold cross-validation at seeds 11, 23, 37, 41, and 59. Every task is
held out exactly once per seed and no repository crosses a fold. Apply these gates independently
to every seed:

1. at least 95 percent quality retention versus that seed's strongest static arm;
2. at least 40 percent realized cost savings;
3. positive matched task-blind reward advantage at identical arm traffic;
4. no static arm with equal or greater reward at equal or lower cost;
5. complete task coverage with no reward-dependent dropping.

For every point passing those gates, fit 128 deterministic null policies. Each null permutes
complete six-arm outcome rows as repository blocks inside exact language-profile and repository
task-count strata, using seeds `20260801` through `20260928`. Features, folds, context fitting,
budget, rounding, and actual held-out evaluation remain unchanged. A point passes the null gate
only when its mean five-seed matched-blind advantage exceeds the higher empirical 95th percentile
of the 128 null advantages and real-minus-null95 is positive in at least four seeds.

Select the passing candidate with lowest mean realized cost, then highest worst-seed retention,
highest real-minus-null95 advantage, fewer contexts, smaller hash dimension, larger shrinkage, and
frozen grid order. Report WISERouter SingleBest, cost-only allocation, task-blind allocation,
graded kNN, graded IRT, every static arm, all 15 pair oracles, and the full oracle.

## Latency and persistence

Audit every scientifically eligible point on one no-internet E2B CPU for at least 10,000 exact
decisions. The measured path includes task hashing, prompt-shape construction, standardization,
nearest-centroid assignment, adaptive workload-budget update, linear-program resolution, and
deterministic rounding state. It must make zero network calls and remain below 5 ms p50 and 20 ms
p95. A missing latency row fails promotion.

All fitting, null evaluation, bootstrapping, and latency work run on E2B or Azure, never the Mac.
Only aggregate metrics, input and source hashes, and conditionally frozen label-free routes return.
Persist no centroids, context statistics, task vectors, outcome matrix, fitted numeric policy, or
foundation model. Destroy every sandbox and prove each exact identifier unavailable afterward.

## External confirmation and target transfer

If development passes, refit the selected point ephemerally on all development rows and freeze one
label-free route for every confirmation task before opening confirmation outcomes. Run the frozen
dense six-arm confirmation exactly once so all static arms, all pair oracles, matched-blind
traffic, and the full oracle remain gradeable.

Confirmation uses the existing 10,000 repository-cluster bootstrap gates: at least 95 percent
quality retention, at least 40 percent savings, nonnegative paired quality lower bound, strictly
positive matched-blind lower bound, no static dominance, at least 95 percent whole-task coverage,
and route p95 below 20 ms. A failed confirmation is final for this candidate family.

Only a policy passing confirmation may freeze 113 label-free DeepSWE routes and receive the single
authorized target evaluation. Never tune, repair, or rerun from confirmation or target outcomes.

## Spend and stop rule

Development fitting makes zero provider calls. Rough cumulative experiment spend begins at USD
4,135.54607635. The user authorized a USD 20,000 hard ceiling and monitors provider billing
externally. If development fails, confirmation and DeepSWE remain sealed. If it passes, the dense
six-arm confirmation is the only next paid tranche.

## Final development result

The frozen development study ran once on a secure, no-internet, single-CPU E2B sandbox from
commit `5a82903a`. It completed 90 preregistered candidates over the audited 649-task matrix with
zero provider calls. Sixty candidates produced complete five-seed metrics. All 30 candidates with
32 contexts failed closed because at least one fit context fell below the frozen support minimum.

No candidate passed the primary development gates, so the null study, latency audit, confirmation
route freeze, external confirmation, and DeepSWE target evaluation did not run. The closest point
to the quality gate was `hash2048-j8-shrink0-save0.4`: its worst seed retained 84.42 percent of
static quality and saved 38.16 percent, with a positive 0.00980 matched-blind advantage. The point
with the greatest worst-seed savings was `hash2048-j8-shrink4-save0.6`: it saved 58.09 percent but
retained only 76.32 percent quality. The workload-budget family therefore does not satisfy the
frozen 95 percent quality and 40 percent savings objective on external development.

The aggregate report SHA-256 is
`bf4834ca16141dd940c3b5db192a01c3f7a1338287cfb7c3145bbc6246dbd0e4`. E2B sandbox
`i1n9f3okwggj5lwwb79t6` was destroyed and an exact-ID active-sandbox check found it unavailable.
The report confirms zero target or confirmation outcome access, zero persisted fitted numeric
state, zero persisted task vectors or outcome matrix, and unchanged rough cumulative spend of USD
4,135.54607635.
