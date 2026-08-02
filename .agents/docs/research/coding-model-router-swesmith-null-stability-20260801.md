# SWE-smith null-penalized difficulty routing protocol

Status: development failed. No confirmation was authorized. DeepSWE outcomes
remain sealed.

## Objective

Learn a deterministic pre-inference coding-task difficulty score from external
SWE-smith trajectories, use it to route `gpt-5.6-luna` reasoning effort, and
advance to one DeepSWE transfer only if a new untouched external confirmation
proves task signal beyond a multi-permutation null distribution.

This is a new candidate family after the SWE-rebench guarded-kNN route failed
its shuffled-label negative control. It may use that failure mode to strengthen
the protocol, but it may not reuse confirmation outcomes to choose features,
arms, thresholds, or a target route.

## External training source

The training source is `SWE-bench/SWE-smith-trajectories` at revision
`08e109b4a59eaeebf80e4675cd125d42e7ac99a4`. Only the `tool` split is used.
It contains 24,100 trajectories across three recorded teacher models. The
eight frozen Parquet LFS object hashes are:

1. `ac76e9efe75978f83d4daec98a55604587dd58b0f1509364893b8778cbf5b487`
2. `2d3a4fcdb89bf4ec4485a1abc41d92645a446d115705cbe882dd5fe148afec3a`
3. `13a95587cfc2fac8e2d6130b25167c02362add1cf969888574328a49dc8ad319`
4. `677831f640fb14e5b7bc02fd2aa729c05cd2068750777eaf877084de03157594`
5. `e41ffbd3f9d1571917ac5b9f4d0108ded4799d9776f597a054203467135a5715`
6. `487a3a6bc8d00c4d2e6ace23da10a82c03349ada8b0f6758d5abf8c7f7fa61d4`
7. `c20d2fc45e67af5d5c6d4c91ca7dfeac69cbaf40442bc2bfe7bac528a99e4b59`
8. `29b684a990ec25345e9c7ade1b8287a6e41fa7fbabd4734f45420ed29280c868`

Preparation reads only `messages`, `instance_id`, `resolved`, `model`, and
`traj_id`. From `messages` it retains only the initial user task text. It never
uses later assistant reasoning, tool actions, observations, patches, or hidden
tests as routing features. Repeated trajectories are aggregated by task.
Task IDs with more than one whitespace-normalized problem statement are
excluded rather than merging prompt-version labels. This source-integrity rule
was frozen after the first two isolated preparation attempts found such a task,
before any retained difficulty result or aggregate label statistic was opened.

The smoothed external difficulty target is the beta-binomial posterior mean
`(resolved + 1) / (trials + 2)`. A task needs at least three distinct
trajectories. Repositories or normalized exact prompts overlapping DeepSWE are
dropped. Repositories in the existing SWE-rebench development cohort are also
dropped before difficulty fitting so calibration cannot retrieve a training
sibling.

All download, parsing, fitting, permutation work, and route generation run on
E2B. The Mac stores only compact manifests, audits, reports, and label-free
route decisions. No foundation model, trajectory corpus, or fitted numeric
router state is persisted.

## Candidate family

The external difficulty scorer uses only the initial task text and canonical
repository string. Candidates are:

- signed character 3-to-5-gram hashing at 512, 2,048, or 8,192 dimensions;
- Ridge alpha 1, 10, or 100;
- a fixed structural feature block containing prompt characters, lines, code
  fences, stack-trace markers, path tokens, quoted identifiers, issue verbs,
  test terms, dependency terms, and language hints;
- the same structural block alone as a low-capacity control.

Every scorer is evaluated with five repository-grouped folds on SWE-smith.
Selection minimizes grouped mean squared error, then maximizes grouped rank
correlation, then chooses lower dimension and larger regularization. The
selected scorer is fit ephemerally on all eligible external tasks and produces
only a difficulty score for each label-free calibration or confirmation task.

The effort policy is monotone over difficulty and may route only between the
`luna-high` and `luna-max` arms selected before the failed confirmation was
opened. Threshold candidates are the 10th through 90th percentiles of the
development difficulty score in 5-point increments. The policy routes harder
tasks to `max` and all others to `high`. Lower predicted external success means
harder, so a task routes to `max` when its score is at or below the threshold.
No confirmation result may change the scorer, arm set, threshold grid, or
direction.

## Null-penalized development selection

Effort calibration uses only the existing SWE-rebench development matrix and
its repository groups. It does not use either prior confirmation outcome.

For each threshold, compute repository-grouped five-fold out-of-fold reward,
cost, quality retention, static dominance, and matched task-blind advantage.
Then generate 128 deterministic null scorers by permuting whole repository
difficulty-score blocks within exact language and retained-task-count strata,
using seed `20260801`. Repositories in this cohort contain one to three retained
tasks, so this moves complete equal-length blocks without splitting, padding, or
interpolation. Within-repository task order stays fixed. Each null route keeps
the candidate threshold and arm semantics. The null-95 value uses the higher
empirical quantile.

A candidate is eligible only when:

1. every development fold retains at least 95 percent of its strongest static
   reward;
2. aggregate matched task-blind advantage is positive;
3. no static effort has at least its reward at no greater cost;
4. real matched-blind advantage exceeds the 95th percentile of the 128 null
   advantages;
5. real minus null-95 advantage is positive in at least four of five folds;
6. its threshold is selected in at least four of five leave-one-repository-fold
   stability refits; and
7. complete route latency p95 is below 5 ms on the E2B reference worker.

Among eligible candidates, select the lowest cost, then largest real-minus-null
margin, then higher reward, then the frozen candidate order. If no candidate is
eligible, the study stops without a paid confirmation.

## New external confirmation

Only an eligible development candidate authorizes a new 200-task confirmation
cohort from the same pinned SWE-rebench source revision. Label-free selection
uses seed `20260801`, at most three tasks per repository, and the existing
60 Go, 10 JavaScript, 60 Python, 10 Rust, and 60 TypeScript quotas.

The cohort excludes every repository in:

- DeepSWE;
- the prior SWE-rebench development and confirmation cohorts; and
- the retained SWE-smith difficulty-training corpus.

It also excludes normalized exact-prompt overlap with DeepSWE, either prior
SWE-rebench cohort, or SWE-smith training. If the frozen quotas cannot be filled
after these label-free exclusions, this study stops before provider execution.
It does not relax quotas or substitute a post-outcome cohort.

The confirmation matrix uses the same model, five efforts, two attempts,
harness, runtime, scoring, timeout, telemetry, whole-task exclusion, and retry
rules as the completed SWE-rebench experiment. Confirmation routes are frozen
before any confirmation outcome exists.

Confirmation passes only when:

1. retained task coverage is at least 95 percent and every retained cell is
   gradeable;
2. router-minus-matched-blind reward has a repository-bootstrap 95 percent
   lower bound above zero;
3. router-minus-the-best-of-128-frozen-null-routes has a repository-bootstrap
   95 percent lower bound above zero;
4. the router retains at least 95 percent of the strongest static reward;
5. no static effort dominates it;
6. route p95 remains below 5 ms; and
7. the isolation audit reports no target outcome access or repository and
   exact-prompt overlap.

A failed confirmation is final for this candidate. Confirmation-dependent
tuning or a replacement confirmation cohort is prohibited.

## DeepSWE boundary and budget

Only a fully passing new external confirmation can authorize the existing
single hash-pinned DeepSWE evaluation. The target matrix remains
`2988742e48b1c9bfec8dc45d88af112c46c45367529d1936b709e4b4e549835f`.
Target decisions must be frozen on a no-internet E2B worker before the matrix is
opened. There is no target rerun or target-dependent tuning.

Rough cumulative spend before this study is USD 887.541861 against the user
authorized USD 20,000 ceiling. Exact metering remains non-blocking, but every
paid tranche must preserve exact counters when available and otherwise record a
clearly labeled trace-derived estimate.

## Terminal development result

The compact source preparation completed on E2B from all 24,100 frozen
trajectories. It retained 1,551 tasks across 115 repositories after excluding
873 task IDs with prompt-version ambiguity, 8,012 tasks with fewer than three
trajectories, and 201 repository overlaps. The compact task artifact SHA-256 is
`9a4b3b749fb2123933335f9c4db41057247f49b37c53a7c075143b44e800aa7c`.
The source sandbox was terminated and no trajectory corpus or fitted model was
persisted.

The no-internet E2B fit selected `charhash8192-a10` on external grouped CV. Its
external grouped MSE was 0.05342377 and grouped Spearman correlation was
0.105776. No effort threshold passed the frozen null-stability gates. The best
apparent development point was the 45th-percentile threshold at 0.510256 reward,
0.995 quality retention, USD 0.162296 per task, and 0.026982 matched-blind
advantage, but its advantage exactly matched the higher 95th percentile of the
128 repository-block nulls. Four of five leave-one-fold refits had no eligible
threshold, and the fifth chose a different single threshold. Route latency p95
was 4.9161115 ms.

The terminal report SHA-256 is
`42672d883940f82d42290820ec2fa7bbd3dcd404cc86d75aa4dea9a7a2f09421`.
The fit sandbox was terminated, no numeric router state was persisted, no paid
confirmation was launched, and DeepSWE was not opened.
