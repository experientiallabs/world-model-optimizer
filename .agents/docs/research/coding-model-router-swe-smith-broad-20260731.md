# Broad SWE-smith difficulty-transfer experiment

Date: 2026-07-31

Status: complete negative external result, DeepSWE outcomes unopened

## Result

The fit-only consensus was `hash512-ridge-heads-a1`, a 512-dimensional signed hashing transform
with two Ridge reward heads. It was feasible on all five fit partitions, with mean fit retention
0.9519 and mean strong traffic 0.7684. All five consensus artifacts passed independent-refit route
parity, zero-network, and 10,000-decision latency audits. Artifact size was 8.8 KB. Observed p50
latency was 1.19 to 2.33 ms and p95 was 3.74 to 5.79 ms.

The single frozen outer-heldout replay did not promote. Per-seed retention was 0.9388, 0.9614,
0.9363, 0.9845, and 0.9351. Strong traffic was 0.4792, 0.5000, 0.7333, 0.6667, and 0.6038, so the
traffic gate passed but the retention gate failed in seeds 0, 2, and 4. Pooled retention was
0.9537 with a repository-bootstrap 95 percent interval of `[0.9161, 0.9921]`, whose lower bound
failed the 0.95 gate.

The router beat matched task-blind mixing by 0.0281 reward with a 95 percent interval of
`[0.0067, 0.0560]`. It beat uniform random and weak static too, but did not beat the selected
shuffled-label controls: the router-minus-shuffled interval was `[-0.0082, 0.0205]`. Three
repository catastrophes also failed the frozen safety gate: `conan-io/conan` lost 0.1250,
`facebookresearch/hydra` lost 0.1331, and `pydicom/pydicom` lost 0.1567 against strong static.

The selection lock SHA-256 is
`e838c39eed0298fa484e80c30ea4bc17fc7cdbb63730278875b6b5e9a96776a8`. The immutable heldout
rows SHA-256 is `2f087b25e2da9fa44928ece6fb04ed63603af8748d0434332a184b6a8d37c6da`, and the promotion
report SHA-256 is `78c5a02b1d63f329e8d665f35d31f05339bedf71d6fcb3b23eae505c28277915`.
The promotion report proves `target_outcomes_used: false`. Per the frozen protocol, this negative
result forbids a DeepSWE replay and cannot be repaired by tuning against the opened source
heldout rows.

## Question

Can a latency-neutral prompt router learn which software-engineering tasks need the stronger of
two externally observed coding agents, then transfer that task-difficulty rule to
`gpt-5.6-luna` reasoning effort on DeepSWE v1.1 without fitting on DeepSWE outcomes?

This experiment is a response to two valid negative findings. BigCodeBench provided a dense
current-model effort matrix, but its corrected held-out-attempt oracle found only 0.0166 absolute
headroom over the fit-selected static effort, far below the frozen 0.10 gate. The first paired
SWE-smith-rs candidate retained 1,372 tasks but only eight repositories, so its labels remained
unopened. The broader source below has enough independent repositories for grouped validation.

The served policy may contain only deterministic text features and small fitted numeric arrays.
It makes no model call, stores no foundation-model weights, and does not inspect an answer,
trajectory, verifier output, or target outcome before routing.

## Frozen external sources

Trajectory source:

- dataset: `SWE-bench/SWE-smith-trajectories`
- commit: `08e109b4a59eaeebf80e4675cd125d42e7ac99a4`
- prompt serializations: `xml` and `ticks`
- weak arm: `claude-3-5-sonnet-20241022`
- strong arm: `claude-3-7-sonnet-20250219`

Canonical pre-call task text:

- dataset: `SWE-bench/SWE-smith`
- commit: `ea6d7173829c7ec8fa16c22055699ff2e9188091`
- split: `train`

The arm order is frozen by model generation, with Claude 3.7 treated as the stronger successor to
Claude 3.5. No outcome statistic selected the pair. The `xml` and `ticks` cohorts were selected
because they have the same 546 exact paired task identities, while `tool` has only 471. No reward
column was read to make this choice.

An earlier feasibility inspection saw aggregate model frequencies for all three prompt formats.
No task-level outcome from this broad source was read before this protocol freeze. The separate
SWE-smith-rs feasibility pass had already exposed aggregate resolve counts and the first ten
outcomes from different datasets. Those observations do not enter this source's cohort, arm
choice, split, feature, or threshold selection.

## Label-free cohort freeze

The label-free freeze has SHA-256
`96427a4e3f8db70ff661dece0459da04006869247d7f8cdefd684c82f898c939` and lives in the ignored
experiment artifact tree as `swe-smith-v32-label-free-freeze.json`.

The cohort is the intersection of tasks that have both frozen model arms in both frozen prompt
serializations and have canonical task text. Eight of 546 paired identities are absent from the
pinned canonical task table, leaving 538 tasks across 72 canonical repositories. Exact task-id,
normalized-prompt, and canonical-repository overlap with the label-free DeepSWE feature view are
all zero. The cohort identity SHA-256 is
`344d533822da71a73338fc534041b8407e02bac8b46d66e41a7bf721d02356f5`.

Repository identity is the lowercased `owner/repo` prefix of the SWE-smith instance id. All rows
for one task and repository remain in one fold. Five identity-only outer splits were frozen before
opening source rewards:

| Seed | Fit tasks | Held-out tasks | Fit repos | Held-out repos | Held-out digest |
|---:|---:|---:|---:|---:|---|
| 0 | 442 | 96 | 54 | 18 | `035118f773817fe54fa4b653ef0a627ec862c7ffb5e6fbdd1a617df57f5514e3` |
| 1 | 450 | 88 | 63 | 9 | `d16b8ee63ae2a01416315250cfcc41681abdae9c8a952b86f3854c483cbeded1` |
| 2 | 433 | 105 | 57 | 15 | `22b3bbb3be1d98532a230c6ae5c871c858b5310906bfe0d01d18a04a3bd0c985` |
| 3 | 421 | 117 | 57 | 15 | `b3aba71ecb473044d5037e9ddcbf1d8269d43300e4c42fe9718515af21d76acc` |
| 4 | 432 | 106 | 57 | 15 | `4995fef8f029b35a50b60175df9f581cfa5119833f316f62903f61125b629d14` |

For each seed, repositories are ordered by
`sha256("swe-smith-broad-outer-v1:<seed>:<repo>")`. The held-out prefix is the count immediately
before or after 20 percent that is closest to the target, with the longer prefix winning an exact
tie.

## Outcome materialization

Only after this protocol is committed may the runner read `resolved`. It must verify the frozen
dataset commits, native shard inventory, task identities, prompt hashes, cohort digest, and all
five held-out digests before writing a compact paired source.

Every gradeable trajectory remains an attempt. For each task and model:

1. average all attempts within `xml`;
2. average all attempts within `ticks`;
3. average the two serialization means with equal 0.5 weight.

This prevents a task with extra archived trajectories from receiving more weight and treats
prompt serialization as a nuisance variable. Every included task must have at least one gradeable
row for both arms in both serializations. Missing or nonbinary rewards invalidate the task rather
than silently changing its arm or serialization weights. The report includes raw row counts,
attempt-count distributions, per-arm means, source hashes, and the exact dropped-task list.

This source has no comparable measured inference cost contract. External selection therefore
optimizes strong-arm traffic at a quality constraint and makes no source cost-savings claim.

## Held-out-format oracle gate

Before fitting a task-text router, measure whether task-specific arm preference is stable across
the two prompt serializations:

1. On `xml`, choose the better arm per task and the better static arm globally.
2. Score both choices on `ticks`.
3. Reverse fit and held-out serializations.
4. Average the two direction estimates.
5. Run 5,000 deterministic canonical-repository bootstrap resamples with seed `20260731`.

Proceed only when every condition passes:

1. at least 500 tasks and 50 repositories remain;
2. both arms and both serializations are dense on the same cohort;
3. the strong static arm exceeds the weak static arm and its paired 95 percent lower bound is
   above 0.01 absolute reward;
4. mean held-out oracle headroom over the fit-selected static arm is at least 0.03;
5. the oracle headroom 95 percent lower bound is above 0.01;
6. no target outcome was read.

If the oracle fails, preserve the negative result and do not fit or replay DeepSWE.

The oracle passed after the protocol commit. It retained 538 tasks across 72 repositories. The
strong-minus-weak reward difference was 0.17530 with a 95 percent interval of
`[0.12306, 0.23631]`. Mean cross-format oracle headroom was 0.04931 with a 95 percent interval of
`[0.02356, 0.07321]`. The paired source SHA-256 is
`3ebc3cbf2ee69e6220b9751193b7c148e6f65d7dca2588df4943b9a66f722553`.

## Frozen router search

If the oracle passes, evaluate this union of previously implemented CPU candidate families:

1. signed hashing Ridge reward heads at 512, 2,048, and 8,192 dimensions with alpha 0.1, 1,
   and 10;
2. word and character TF-IDF plus SVD Ridge uplift and reward heads from the frozen `full`
   autoresearch family;
3. word TF-IDF plus SVD ExtraTrees reward heads with minimum leaf size 5 and 20;
4. word TF-IDF plus SVD histogram-gradient reward heads with minimum leaf size 10 and 30;
5. deterministic 27-dimensional prompt-shape Ridge reward heads and the four frozen structural
   IRT regularization points;
6. WMO guarded local kNN using hashing dimensions 512, 2,048, and 8,192, neighbors 8, 16, 32,
   and 64, relative similarity 0.90, 0.95, and 0.98, guard z 0, 0.5, 1.0, and 1.645, and minimum
   paired support 8, 16, and 32.

Negative controls are task-blind uplift, within-fit-repository shuffled outcomes, deterministic
uniform random traffic, weak static, strong static, and the unattainable per-task oracle. Features
use canonical pre-call problem text only. Repository identity, source trajectory content, model
answer, test names, patch, and verifier details are excluded from the policy features.

Within each outer-fit partition, use five repository-grouped inner folds. Candidate and threshold
selection uses inner out-of-fold predictions only. For each candidate, consider the thresholds
that retain 0.95, 0.97, and 0.99 of the inner-fold strong static quality. Select the point with the
least strong-arm traffic among those retaining at least 0.95 quality. Ties use higher reward,
smaller route latency, smaller serialized artifact, then frozen family and grid order. If no point
is feasible, keep the highest-quality point but mark that seed infeasible.

Persist one immutable fit report and one selected artifact audit per outer seed before any outer
held-out replay. The audit reloads the artifact, proves route parity, proves zero network calls,
and measures at least 10,000 decisions on one E2B CPU core. Required latency is below 5 ms p50 and
20 ms p95.

The five fit-only searches ran from commit
`ed0cd24df9d8db58a3639e84de099d2fe392ded1` on isolated, internet-disabled E2B CPU workers. Each
report contains 459 rows: 455 eligible candidates and four ineligible negative controls. The
per-seed mechanical winners were:

| Seed | Winner | Strong traffic | Reward | Strong reward | Retention | Report SHA-256 |
|---:|---|---:|---:|---:|---:|---|
| 0 | `hash512-ridge-heads-a1` | 0.7443 | 0.4310 | 0.4519 | 0.9537 | `2026d83b4e1b6c5e9299972d8028d278b11089f500eb0090cb792443e19dbbf1` |
| 1 | `hash512-ridge-heads-a1` | 0.7444 | 0.4774 | 0.5024 | 0.9503 | `a9a802cb7e30a82469ed2bbc2cbc7815e6338e8ecdf682ab951a40b5c710268c` |
| 2 | `hash512-ridge-heads-a0.1` | 0.7921 | 0.4405 | 0.4632 | 0.9510 | `092b9fd8d08ff2fa3f48572a82fb4bc8c2340f54afcb3a61543583fcdedfc5f3` |
| 3 | `hash512-ridge-heads-a1` | 0.7601 | 0.4313 | 0.4537 | 0.9506 | `d46504e967c34dabcba26526ed4786c8798487d25945d2790194ab665a68ff19` |
| 4 | `hash512-ridge-heads-a1` | 0.7292 | 0.4892 | 0.5133 | 0.9530 | `32df145cb603917d7ca63a3e075977476da71c75557496ac6ad1ce85e3a51129` |

All five selected artifacts passed independent-refit route parity and internet-disabled serving
audits over 10,000 decisions. Observed p50 latency was 1.18 to 2.34 ms and p95 was 3.75 to 5.67 ms.
The fit-only consensus lock must still complete before any outer-heldout source file is opened.

## External promotion

Replay each locked seed once on its outer-heldout repositories. Promotion requires:

1. every seed retains at least 0.95 of its fit-selected strong static arm;
2. every seed sends at least 20 percent fewer tasks to the strong arm than strong static;
3. pooled repository-bootstrap retention has a 95 percent lower bound at or above 0.95;
4. pooled router reward minus a task-blind policy with identical per-seed strong traffic has a
   strictly positive 95 percent lower bound;
5. pooled router reward also has a strictly positive lower bound against shuffled-label,
   deterministic-random, and weak-static controls;
6. no repository with at least five held-out tasks loses more than 0.10 absolute reward against
   the fit-selected static arm;
7. all five selected artifacts pass the serving latency and zero-call audit.

Use 10,000 deterministic canonical-repository bootstrap resamples with seed `20260731`. The
matched task-blind control samples strong routes without replacement inside each seed at exactly
the router's held-out strong count. It is not allowed to use target labels or target traffic to
set a threshold.

Before outer-heldout replay, the exact negative-control randomness is frozen as follows. The
matched task-blind subset orders task ids by
`sha256("swe-smith-broad-task-blind-v1:<seed>:<instance_id>")` and takes exactly the router's
strong-route count. Uniform random routing uses the leading digest byte from
`sha256("swe-smith-broad-random-v1:<seed>:<instance_id>")` with a threshold of 128. The selected
shuffled-label numeric control is refit after jointly permuting weak and strong outcome pairs
inside each fit repository with NumPy seed `30000 + outer_seed`. Bootstrap resampling draws the
same number of repositories with replacement independently inside each outer seed, then pools
the retained task rows. These settings cannot change after the first outer-heldout file is read.

Only one fit-selected consensus may advance. A candidate is consensus-feasible only when its
inner-fold point retains 0.95 strong quality in every seed. Among feasible candidates, choose the
lowest mean fit strong traffic, then the tie-breaks above. Refit that exact configuration on all
538 external tasks only after positive external promotion.

## Frozen DeepSWE transfer

DeepSWE v1.1 remains evaluation-only. A positive external promotion authorizes exactly one target
replay:

- source weak route maps to `gpt-5.6-luna` at `xhigh` effort;
- source strong route maps to `gpt-5.6-luna` at `max` effort.

The mapping is frozen because prior DeepSWE frontier analysis identified xhigh and max as the
useful adjacent cost-quality pair. The policy itself remains fitted only on this external source.
The target replay uses graded `f2p_passed / f2p_total`, measured cost, model-times-effort arm
identity, and repository-grouped uncertainty. Report Luna xhigh, Luna max, the frozen router, and
a matched task-blind policy at identical max-effort traffic.

No target feature search, threshold adjustment, representation change, cost-penalty tuning,
candidate swap, or second replay is permitted. The result is adaptive cross-dataset transfer, not
an untouched confirmation set.

## Compute and durability

All source scans, bootstraps, fitting, and latency audits run on E2B or Azure. The local Mac is
limited to code editing, orchestration, compact artifact sync, and lightweight tests. No Modal
application is used. No provider call is required for this source study, and no foundation model
is downloaded or persisted.

Raw source outcomes, compact rows, fit reports, audit artifacts, selection locks, and promotion
reports stay in the ignored experiment artifact tree. Reusable runner changes receive focused
tests, whole-project Ruff and ty gates, a commit, and a push before remote fitting.
