# Repository-tree graded router follow-up

Status: frozen after the graded kNN, IRT, and workload-budget development failures, before source
tree acquisition, feature construction, or fitting. The 320-task external confirmation and every
DeepSWE outcome remain sealed.

## Question

Can zero-LLM repository structure and issue-to-file-path localization features identify when a
cheaper Luna reasoning effort can replace `sol-max`, while preserving a latency-neutral route?

Prompt-only character, semantic, trace-difficulty, direct-uplift, IRT, and workload-context
families have all failed to recover the external pair-oracle headroom. SWE-Router argues that
multi-turn software tasks have an information-theoretic prompt-only routing floor because similar
issue descriptions can hide very different repository work. It resolves that ambiguity with a
partial model trajectory: `https://arxiv.org/abs/2607.00053`. That adds model cost and latency,
which this experiment forbids. This lane instead tests information already present in the checked
out repository before the first model call. Interpretable subgroup discovery is motivated by the
SWE-bench analysis in Where Do Agents Differ:
`https://openreview.net/forum?id=ZROtXcOcLT`.

## Evidence boundary

Development reuses only the audited 649-task, six-arm graded SWE-rebench matrix with outcomes
SHA-256 `5023abf4e16d52a0a324bead44b1db80b518c2408593189585f0a4b416f94822` and completion
audit SHA-256 `d256c23e6661d9b1ef232c74d52922b5c1ce83a69ca1c87e723c97266a71064b`.
The arms remain `luna-low`, `luna-medium`, `luna-high`, `luna-xhigh`, `luna-max`, and
`sol-max`.

Task identity, repository, language, initial issue text, and exact base commit come from
`nebius/SWE-rebench-V2` at immutable revision
`475dd5e8703bb5fb22dd3c60b5d038b019eba1e0`. Its native Parquet SHA-256 is
`0e0bf9355f892ad74ae98d4e1c404f39fd6654a8e351ee3e6ab162e4a64cd3ad`. Join rows only by
exact task identity and reject any prompt, repository, language, or image mismatch with the frozen
development and confirmation manifests.

The only new raw input is the public Git tree at the task's exact base commit. Acquire it through
GitHub's commit-addressed tree API or an equivalent `git ls-tree` over that exact commit. Retain
only path, object type, executable mode, and blob byte size. File contents, blob hashes, commit
messages, branches, later repository state, patches, tests, test identities, verifier data,
install commands, dataset-generated LLM metadata, model output, trajectories, rewards, and costs
are forbidden feature inputs. Gold patch and test patch columns must never be loaded by the
feature worker.

The confirmation manifest remains the unopened 320-task repository-disjoint cohort with SHA-256
`c9443c9956e496123f396ee793efbb3368312092c4dcbd4e5e10bb77bd814f0a`. No confirmation or
DeepSWE outcome may enter source acquisition, fitting, selection, or route freezing.

## Frozen repository feature contract

Normalize paths with Unicode NFKC, forward slashes, and case folding. Reject absolute paths,
parent traversal, empty components, submodule objects, and trees that report truncation. Exclude
paths under exact generated or vendored directory components `vendor`, `vendors`, `third_party`,
`third-party`, `node_modules`, `dist`, `build`, `target`, `.git`, `.tox`, `.venv`, and `venv`.
Do not infer generated status from file contents.

For every valid tree, compute this fixed numeric block:

1. log file count, log total blob bytes, executable-file fraction, and symlink fraction;
2. mean, standard deviation, median, 90th percentile, and maximum path depth;
3. mean, standard deviation, median, 90th percentile, and maximum blob size after `log1p`;
4. fractions of paths classified as source, test, documentation, configuration, examples,
   generated artifacts, or unclassified by frozen path and extension rules;
5. source-to-test, source-to-documentation, and configuration-to-file ratios;
6. extension count, normalized extension entropy, dominant-extension fraction, and language-file
   fraction for the frozen task language;
7. binary markers for Python packaging, Node, TypeScript, Go, Rust, Java Maven, Java Gradle, Bazel,
   CMake, Make, Ruby, PHP, .NET, Julia, Swift, and mixed-language or monorepo layouts;
8. counts of top-level directories, top-level source roots, test roots, package manifests, lock
   files, CI configurations, and distinct build systems; and
9. fractions of paths at depth one, two, three, four, and five or greater.

The exact source, test, documentation, configuration, extension, language, and build-marker tuples
live in the implementation and are immutable after the first tree is acquired.

Build a second task-specific block from the initial issue text and file paths only. Tokenize issue
text and path components into case-folded alphanumeric terms of length at least two. Remove the
fixed English stopword set shipped by scikit-learn. Split camel case, snake case, kebab case, and
letter-to-digit boundaries. Use no learned tokenizer and no corpus-derived stopwords.

For each path, compute BM25 with `k1=1.2` and `b=0.75` over path-component tokens, with inverse
document frequency fit only on that task's repository paths. Record maximum, mean, standard
deviation, 90th and 99th percentile BM25; top-one minus top-two margin; top-one divided by the sum
of positive scores; count and fraction of paths with positive score; and normalized entropy of
positive scores. Repeat maximum, margin, concentration, positive count, and entropy for source
paths and test paths separately. Add exact counts for issue backtick spans, slash-containing spans,
filename-like spans, and how many of those spans match a complete path, suffix, basename, stem,
extension, or directory component. Add issue-token coverage by the union of path tokens and the
number of distinct top-level directories represented in the top 1, 3, 5, and 10 BM25 paths.

Append the existing frozen 15-value prompt-shape block only as a required ablation coordinate.
Standardize every continuous feature inside each fit fold. Replace no missing value with an
outcome-derived statistic: use zero plus a missingness bit for undefined ratios and empty path
subsets. Reject non-finite feature rows.

Tree acquisition and features are scientifically eligible only if they cover at least 95 percent
of the 649 development tasks before outcomes are joined. Missing or invalid trees exclude the
whole task from every arm without an outcome-dependent retry. Report coverage by language and
repository size. Do not relax the frozen feature contract to improve coverage.

## Frozen direct substitution family

The guard is fixed to `sol-max`. Each Luna effort is considered separately as the cheap arm. For
task `i`, the supervised target is the complete graded difference
`reward[sol-max] - reward[cheap-arm]`. Costs never enter this target.

Compare exactly three feature views:

1. repository structure only;
2. repository structure plus issue-to-path localization; and
3. repository structure plus issue-to-path localization plus prompt shape.

The only learner is `HistGradientBoostingRegressor` with squared-error loss, learning rate 0.05,
maximum 200 iterations, and early stopping disabled. Cross these frozen coordinates:

- maximum leaves: 7, 15, or 31;
- minimum leaf size: 10, 25, or 50; and
- L2 regularization: 1 or 10.

This gives 18 structures per feature view and 54 structures per cheap arm. The complete family has
270 structures. For each out-of-fold score vector, thresholds are score ranks 20, 25, through 80
percent. Tasks at or above the threshold route to `sol-max`; all others route to the paired cheap
arm. The family therefore has 3,510 deterministic operating points before null controls. Ties use
lower expected cost, higher predicted Sol-minus-cheap reward, then frozen task identity order.

Use repository-grouped five-fold cross-validation at seeds 11, 23, 37, 41, and 59. Every task is
held out exactly once per seed and no repository crosses a fold. Fit all preprocessing inside the
training fold. The first scientific fit freezes the exact scikit-learn version, source hashes,
feature dimension, and candidate count. Any mismatch fails closed.

## Development gates and family null

A real operating point passes the primary gates only if every seed independently has:

1. at least 95 percent quality retention versus `sol-max`;
2. at least 40 percent realized cost savings;
3. strictly positive matched task-blind reward advantage at identical arm traffic;
4. no static arm with equal or higher reward at equal or lower cost; and
5. complete routes over every retained feature row.

For each of 128 null indices with seeds `20260801` through `20260928`, permute complete six-arm
outcome rows as repository blocks within exact language and repository-task-count strata. Refit
all 270 structures, replay all thresholds, and retain the largest mean five-seed matched-blind
advantage among points that satisfy the same quality, savings, dominance, and coverage gates.
This is a best-of-family null that corrects the sequential learner, feature-view, cheap-arm,
structure, and threshold search.

A real point passes the family null only when its mean matched-blind advantage exceeds the higher
95th percentile of the 128 null maxima and its real-minus-null95 seed margin is positive in at
least four seeds. Select the passing point with lowest mean realized cost, then highest worst-seed
quality retention, largest real-minus-null95 advantage, smaller feature view, shallower tree,
larger minimum leaf, larger L2 penalty, and frozen grid order.

Report every static arm, all 15 pair oracles, full oracle, cost-only traffic, matched task-blind
traffic, repository-structure-only ablation, localization ablation, prompt-shape ablation, graded
kNN, graded IRT, and the failed workload-budget frontier.

## Latency and persistence

All tree acquisition, feature construction, fitting, null evaluation, bootstrapping, and latency
measurement run on E2B or Azure, never the Mac. Acquisition may use internet access only for the
exact public Git trees. Fitting, nulls, and latency run in separate no-internet workers.

For every scientifically eligible point, audit at least 10,000 exact decisions on one E2B CPU.
Report two paths separately:

1. cold repository extraction through `git ls-tree` plus feature construction and decision; and
2. warm decision from an in-memory path list plus feature construction.

The promoted route must make zero network calls and remain below 20 ms p95 on both paths. Its warm
p50 must remain below 5 ms. A missing latency row or a repository whose cold path exceeds the p95
gate fails promotion. Environment provisioning and repository checkout are outside the route, but
no task-specific feature cache may be loaded before the cold timer starts.

Only aggregate metrics, source and input hashes, coverage diagnostics, latency aggregates, and
conditionally frozen label-free routes may leave remote compute. Persist no Git tree, file path
list, per-task feature row, model file, coefficients, task embedding, outcome matrix, or fitted
numeric state. Destroy every sandbox and prove each exact identifier unavailable afterward.

## External confirmation and target transfer

If development passes, refit the selected point ephemerally on all retained development rows and
acquire the same label-free tree features for the frozen 320-task confirmation cohort. Freeze one
complete confirmation route, matched task-blind traffic, and 128 repository-block null routes
before opening confirmation outcomes. Destroy the acquisition and fit workers before any provider
call.

The existing dense six-arm confirmation then runs exactly once. Promotion requires at least 95
percent whole-task coverage, at least 95 percent quality retention, at least 40 percent savings, a
nonnegative repository-bootstrap paired quality lower bound, strictly positive lower bounds versus
matched task-blind traffic and the strongest frozen null route, no static dominance, and both cold
and warm route p95 below 20 ms.

Only a policy passing every confirmation gate may acquire label-free base-commit trees for the 113
DeepSWE tasks, freeze target routes, destroy fitted state, and receive the single authorized target
evaluation. Never tune, repair, or rerun from confirmation or DeepSWE outcomes.

## Spend and stop rule

Development tree acquisition and fitting make zero provider-model calls. Rough cumulative
experiment spend begins at USD 4,135.54607635. The user authorized a USD 20,000 hard ceiling and
monitors provider billing externally. If source-tree coverage is below 95 percent, no development
point passes, or latency fails, confirmation and DeepSWE remain sealed and this family stops.

## Source acquisition launch audit

The first remote source-acquisition command downloaded and hash-verified the pinned native
Parquet, then stopped before any Git tree request because task `keras-team__keras-19955` had a
problem statement that did not byte-match the frozen development manifest. The exact E2B worker
was terminated. It made zero provider calls, joined no outcomes, persisted no feature row, and
accessed neither confirmation nor DeepSWE.

The source validator now applies the already frozen rejection rule at whole-task scope. A missing
row or repository, language, prompt, or image mismatch is recorded as a label-free source
exclusion before tree retrieval instead of aborting the complete acquisition command. The 95
percent coverage denominator remains all 649 retained development tasks. Duplicate dataset
identities, source hash drift, schema drift, and any malformed projected field still abort the
complete command. This correction does not normalize a prompt, accept a mismatch, change a
feature, inspect an outcome, or retry a scientific task. It was frozen before the first Git tree
was acquired.

The corrected whole-task rejection run found that only 281 of 649 tasks exactly matched all
frozen source fields. Its then-current controller still acquired those 281 Git trees before
applying the final coverage gate. Of them, 264 produced valid frozen feature rows and 17 failed
tree validation or retrieval. The worker was terminated, no feature row left remote compute, and
no fit began. Since source identity alone capped possible coverage at 43.3 percent, the tree
results cannot change the scientific decision.

The final audit command moved the source coverage gate before every Git request. It reproduced
281 exact source rows, 368 whole-task source exclusions, and 43.297 percent maximum coverage over
the 649-task denominator. It then stopped before tree retrieval or feature construction. Coverage
report SHA-256 is `485579a89d5ce790b093adef4d10c4f74ad2b7601468af5f510f710a8dbc83bb`.
Exact E2B sandbox `ij1uukngkzwx1imp0ptkj` was terminated and returned
`SandboxNotFoundException` on an exact-ID check. All acquisition attempts made zero provider
calls, joined no outcomes, and accessed neither confirmation nor DeepSWE.

This family therefore fails the frozen development coverage gate before fitting. The 270 learner
structures, 3,510 operating points, family null, latency audit, confirmation route freeze,
external confirmation, and DeepSWE target evaluation did not run.
