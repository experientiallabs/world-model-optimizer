# Pooled cross-attempt reasoning-effort uplift protocol

Status: terminal negative. Development passed, but the fresh external
confirmation failed. DeepSWE outcomes remain sealed and this candidate cannot
advance.

## Objective

Learn when `gpt-5.6-luna` at `max` reasoning effort adds value over `high` from
external coding outcomes, while preserving a pre-inference single-call route
under 5 ms. Advance only if nested repository-held-out evaluation beats a
family-wise repository-block null. This is a distinct candidate family after a
generic SWE-smith difficulty threshold failed its null-stability gate.

## Development pool

The development pool combines the two completed SWE-rebench V2 cohorts. The
old confirmation cohort loses all confirmatory status and is development data
only for this new study. A future claim requires a new untouched cohort.

Frozen inputs are:

| Input | SHA-256 |
|---|---|
| Old development tasks | `7d846b5576d15e68fd18ac21bfe0610cc1614b3b35ec0ae0cb8cfae0b82962c1` |
| Old development outcomes | `5c2097116b03291f20bc33d6a376cb01d9a2e9fb182f46c482df5508b7140ee2` |
| Old development audit | `ca20ebdc85bda0482e9c726a95c4216bc1b4acec63fe86db88ef4fc4431ab316` |
| Old confirmation tasks | `9798dd1e58be0d13331d097307670dc3fc3760ad211da20e6367666523f080a7` |
| Old confirmation outcomes | `0c03bcbd935c0983c9e6355413222fb1545206d6ae5a91329b505f77f35300d6` |
| Old confirmation audit | `cced3491bfb5e4cb5eeaebde6473ad399c3ed4ebf8cbee916fb598625a0f4744` |

The retained pool has 393 tasks, 100 development repositories plus the
disjoint old-confirmation repositories, five efforts, and two independent
attempts per task and effort. The learner may use only the `high` and `max`
cells. The other efforts remain static-frontier comparators and may not become
training labels.

Gradeable zeros remain outcomes. The five prior whole-task infrastructure
exclusions and two old-confirmation whole-task exclusions remain excluded. No
cell is rerun, and no task is dropped based on reward or cost.

## Independent trace prior

Rebuild the compact `SWE-bench/SWE-smith-trajectories` `tool` split at revision
`08e109b4a59eaeebf80e4675cd125d42e7ac99a4` using the eight already frozen LFS
object hashes. Preparation uses only initial user text, task identity,
repository, teacher model identity, trajectory identity, and resolved status.

Before aggregation, exclude every repository and normalized exact prompt in:

- DeepSWE's label-free view;
- both pooled SWE-rebench development cohorts.

Task IDs with multiple normalized problem statements and tasks with fewer than
three trajectories remain excluded. The smoothed target remains
`(resolved + 1) / (trials + 2)`. This prior is fit ephemerally and contributes
one scalar external difficulty feature. Its coefficients are never persisted.
The frozen prior scorer is `charhash8192-a10`, selected by the terminal
SWE-smith study's repository-grouped external CV. It is refit reproducibly on
the rebuilt corpus without reopening scorer selection.

The pooled exclusion manifest SHA-256 is
`4f05067a44fc69f1e53bb26f79da615a51119740fe9753a3e690434aa9e2b01f`.
The rebuilt compact manifest SHA-256 is
`ca253f77f7991d4b3ade6d299c6b41c42b7291b077b5d319167d95a73d526939`.
It retained the same 1,551 tasks and 115 repositories as the prior corpus, with
task artifact SHA-256
`9a4b3b749fb2123933335f9c4db41057247f49b37c53a7c075143b44e800aa7c`.
Therefore none of the retained trace-prior repositories overlaps the added old
confirmation cohort.

## Pre-inference features

Every candidate receives exactly:

- canonical repository plus initial task text;
- signed character 3-to-5-gram hashing at 512, 2,048, or 8,192 dimensions;
- the fixed 15-value prompt-shape block used by the terminal SWE-smith study;
- the ephemerally fitted SWE-smith external difficulty score; and
- deterministic interactions between that score and the 15 prompt-shape
  values.

No source code scan, generated description, embedding API, model call, future
trajectory, patch, verifier feedback, hidden test, old-confirmation status, or
DeepSWE outcome may enter a feature.

## Frozen candidate family

The route may choose only `luna-high` or `luna-max`. Its score estimates the
incremental reward of `max` over `high`; larger scores route to `max`.

Candidates are:

1. direct-uplift Ridge at each hash dimension and alpha 1, 10, or 100;
2. two-head Ridge at each hash dimension and alpha 1, 10, or 100, predicting
   high and max reward separately before subtraction;
3. direct-uplift structural-prior HistGradientBoosting with 7 or 15 leaves and
   L2 penalty 1 or 10; and
4. direct-uplift structural-prior ExtraTrees with 256 trees and minimum leaf
   size 5, 10, or 20.

Ridge uses LSQR. Tree seeds are `20260801`. Threshold candidates are the 10th
through 90th score percentiles in 5-point increments. A threshold routes to
`max` at or above the threshold and to `high` otherwise. Candidate and
threshold order above is the final tie breaker.

## Nested cross-attempt evaluation

Use five deterministic repository-grouped outer folds over the pooled tasks.
For each fold, evaluate both attempt directions:

1. fit a candidate on the four training folds using attempt A only;
2. choose its lowest-cost eligible threshold on the same training repositories
   using attempt B only;
3. apply that frozen route to the held-out repositories and score attempt B;
4. reverse A and B, producing ten held-out evaluations total.

Threshold eligibility on the training repositories requires at least 95
percent reward retention versus their strongest static effort, positive reward
advantage over traffic-matched task-blind routing, and no static effort with at
least the reward at no greater cost. When no threshold qualifies, the fold
route is all `max`, which cannot create positive routing evidence.

Aggregate each candidate's ten held-out routes. Compute reward, realized cost,
traffic-matched blind reward and cost, quality retention, absolute reward
difference, static dominance, and per-evaluation matched-blind advantage.

## Family-wise null and promotion gates

For each held-out evaluation, build 128 deterministic score nulls with seed
`20260801`. Permute complete repository score blocks within exact language and
retained-task-count strata, preserving task order and the exact route traffic.
Thresholds, outcomes, and costs stay fixed.

For each null index, compute every candidate's aggregate held-out advantage and
retain the largest value across the complete candidate family. The primary
null-95 is the higher 95th percentile of those 128 family-wise maxima. This
corrects both task-blind alignment and candidate-selection multiplicity.

A development candidate is eligible only when:

1. aggregate reward retains at least 95 percent of the strongest static effort;
2. aggregate matched-blind advantage is positive;
3. no static effort has at least its reward at no greater cost;
4. its advantage exceeds the family-wise null-95;
5. a repository bootstrap 95 percent lower bound for matched-blind advantage is
   above zero, using 10,000 resamples and seed `20260801`;
6. matched-blind advantage is positive in at least seven of ten outer
   evaluations; and
7. complete route latency p95 is below 5 ms on the E2B fit worker.

Among eligible candidates, select lowest cost, then largest advantage over the
family-wise null-95, then higher reward, then frozen order. If no candidate is
eligible, stop without a paid confirmation.

## Fresh external confirmation

Only an eligible development candidate authorizes a new 200-task SWE-rebench
cohort from the same pinned source revision. Label-free selection uses seed
`20260801`, at most three tasks per repository, and quotas of 60 Go, 10
JavaScript, 60 Python, 10 Rust, and 60 TypeScript.

Exclude every repository and normalized exact prompt in DeepSWE, either pooled
development cohort, or the rebuilt SWE-smith trace-prior corpus. Stop if the
exact quotas are infeasible. Do not relax or replace the cohort after outcomes.

Freeze the selected real route and all 128 null routes before provider calls.
Use the same five-effort, two-attempt, official-verifier matrix contract as the
prior SWE-rebench experiment. Confirmation requires at least 95 percent whole
task coverage, dense gradeable retained cells, 95 percent quality retention,
no static dominance, route p95 below 5 ms, and repository-bootstrap lower
bounds above zero for both router minus matched blind and router minus the best
of the 128 frozen null routes.

A failed confirmation is final for this candidate.

## DeepSWE and compute boundary

Only a fully passing fresh external confirmation can authorize the existing
single hash-pinned DeepSWE transfer. Target decisions must be frozen on a
no-internet E2B worker before the target matrix is opened. There is no target
rerun or target-dependent tuning.

All source parsing, fitting, tree construction, null generation, bootstrap
work, and route freezing run on E2B. The Mac may retain only compact inputs,
reports, audits, and route decisions. No foundation model, trajectory corpus,
or fitted numeric router state is persisted. Modal is not used.

Rough cumulative provider spend remains USD 887.541861 against the authorized
USD 20,000 ceiling. This development study makes no provider calls.

## Development result and confirmation freeze

The no-internet E2B development run selected
`direct_ridge-hash8192-a10`. Across the ten nested held-out attempt directions,
it achieved 0.4745547074 reward, 0.9539641944 quality retention versus static
max, USD 0.1486125335 per task, and 0.0156602934 matched-blind advantage. The
repository-bootstrap 95 percent lower bound was 0.0046271447. Eight of ten
held-out evaluations had positive advantage.

The higher 95th percentile of the best-of-25 family null was 0.0148602243, so
the selected route cleared it by 0.0008000690. Its complete shared-feature route
latency p95 was 4.7039182 ms over 1,965 decisions. The in-sample full-pool route
threshold is 0.2033947214. This threshold is a route-freezing input, not
confirmatory evidence.

The development report SHA-256 is
`d168e721a97782915991f7aa92971bc97cf7feb7705242361143d4ce3358bef9`.
The selection lock SHA-256 is
`43e87a80e286ff98f2e1afda4e5db8dbdf231f55c7b458fa27e824f5bbbf8e4e`.
The E2B fit sandbox was terminated and no fitted numeric state was persisted.

The fresh label-free cohort contains 200 tasks across 109 repositories with the
exact frozen language quotas and zero repository, task-id, or normalized-prompt
overlap against DeepSWE, both pooled development cohorts, and the retained
SWE-smith trace prior. Its task SHA-256 is
`6edd8ed4777d6bc48cf29f76a9fb4b9d60e3324908aa79d4d03df8617f6be825`.
Its manifest SHA-256 is
`7bd743a794c5054e053a9d163c088d0f9f72fbd911043c44f90b792801eade60`.
The selection sandbox was terminated before any provider call.

The real route sends 185 tasks to `luna-high` and 15 to `luna-max`. All 128
repository-block null routes were frozen on a no-internet E2B worker before any
provider call. They are distinct, preserve route traffic, and cover all 200
tasks in frozen order. Complete route latency p95 was 4.4168534 ms over 1,000
decisions. No fitted numeric state was persisted, and the route-freeze sandbox
was terminated.

| Frozen route artifact | SHA-256 |
|---|---|
| Real routes | `aac7523746ee9aac0f9789ba9ee4d4e260fad8d2447730102d7aaa44816224c8` |
| Null routes | `4e1570b285eac8da96c13069479f3f7ea9e49b7bedaae8c90d636777a3212a59` |
| Route audit | `4f31fe2245cbad1123beada405d18d12b6e323963bde14c3ac69a90993c4db6b` |
| Freeze lock | `c8deac37d91912e268108a94a227abae34ab858e3bbbd2c637623697da092751` |

The content-addressed execution phase uses 200 isolated E2B workers against the
1,000-sandbox account cap. It runs the five reasoning efforts with two attempts
per task through the official verifier, persists every completed effort, and
does not reuse any prior smoke cell. The one-shot confirmation analysis remains
sealed until collection is complete.

## External confirmation result

The matrix retained 197 of 200 tasks across 108 repositories, or 98.5 percent
coverage, with 1,970 dense gradeable cells. Three tasks from the same Sage
Carbon repository were dropped under the frozen whole-task infrastructure
policy. Two workers exited 137 before durable trace persistence. The third
completed both max-effort inferences with 40 total provider calls, but both
official verifier scoring runs timed out. No scientific cell was rerun. Each
failure sandbox was terminated only after its exact boundary was audited.

The collected outcomes SHA-256 is
`f96a7262de2763a616d93294c54408f9279d33a97cd5d8b31eca6d5de0519be7`.
The completion audit SHA-256 is
`83a8388236d095585c750738b70ecae766260581037da8f6f85bb108301fd6ef`.
The matrix used 30,664 recorded provider calls and cost USD 236.3878768 by the
trace-derived frozen list-price estimate. Rough cumulative experiment spend is
USD 1,123.9297378. Provider usage for the two zero-trace worker crashes is
unavailable and is explicitly excluded from that estimate.

On the retained cohort, the real route sent 182 tasks to `luna-high` and 15 to
`luna-max`. It achieved 0.4543147208 reward at USD 0.1432880030 per task. Its
traffic-matched blind reward was 0.4494060656, an apparent advantage of only
0.0049086552 with repository-bootstrap lower bound -0.0104728827. Quality
retention versus static max was 0.9421052632, below the frozen 0.95 gate.

Frozen null route 113 was strongest after the same whole-task exclusions. It
achieved 0.4746192893 reward at the same retained 182 high and 15 max traffic.
The real route therefore trailed the best of 128 frozen nulls by 0.0203045685,
with repository-bootstrap lower bound -0.0427135678. Coverage, isolation,
latency, and static non-dominance passed. Matched-blind advantage, quality
retention, and best-null comparison failed. The accepted confirmation report
SHA-256 is
`37e2a2915121204bff8782cb301696f3c79a033e5d17b1ba599c67254b7eb553`.

The analysis implementation was audited before accepting the report. Its first
run stopped before producing metrics because retained whole-task exclusions can
change null-route traffic even when full-cohort traffic is preserved. The
comparison was repaired to use the conservative frozen quantity, real reward
minus the highest-reward null on the retained cohort. A subsequent report was
rejected because a reused helper carried seed `20260731`; the protocol-frozen
seed is `20260801`. That invalid report remains preserved separately. The
accepted report above uses 10,000 repository bootstrap resamples with seed
`20260801` for both confidence bounds.

This result rejects the pooled cross-attempt text router. Its small development
margin over the family null did not reproduce, and more of the same task-text
signal is not justified. No DeepSWE transfer is authorized.
