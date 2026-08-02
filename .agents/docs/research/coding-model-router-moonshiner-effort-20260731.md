# Moonshiner reasoning-effort transfer protocol

Status: frozen before provider execution on 2026-07-31.

## Objective

Find a latency-neutral policy that assigns `gpt-5.6-luna` reasoning effort from
task text and cheap structural features. Fit and select the policy only on an
external coding corpus. Evaluate on DeepSWE v1.1 exactly once only after the
external promotion gate passes.

The primary question is whether a task-conditioned policy can allocate `max`
effort more effectively than a matched task-blind mixture over `xhigh` and
`max`. The full five-effort matrix also measures whether `low`, `medium`, or
`high` can create a better source cost-quality frontier.

## Frozen source

- Hugging Face dataset:
  `greghavens/kimi-k3-coding-and-debugging-traces`
- Hugging Face revision:
  `33a874c3affbdb97e142752a9144e6624ef5bd07`
- `dataset-manifest.json` SHA-256:
  `cd579a954d6bc1f5e2c940c7833d84c2194202caf123feb250fc7b4082da61cb`
- Moonshiner repository commit:
  `da981ff7893de29004712c8f4b3f7a414737525e`
- Frozen task corpus SHA-256:
  `2b1a881ad8337369604e778cd770cf9307225224b1474a246e8ac6abc6f09d1c`
- Frozen validation rows SHA-256:
  `902bf905e31818eb3e3dd516be0f85e8409c87d10149c684a94c5f5865c4c69e`

The Hugging Face manifest lists 438 unique task identities. A task enters this
experiment only if its identity occurs in that manifest and the pinned
Moonshiner repository supplies a prompt, offline fixture, deterministic verify
command, protected files, and reference patch. The allowlisted languages are
assembly, Bash, C, C++, JavaScript, Python, TypeScript, and Zsh. Networked,
interactive, oversized, and explicitly tool-behavior tasks are excluded.

The preflight requires the unmodified fixture to fail and the protected-file
preserving reference patch to pass. Thirty-eight tasks met the static selector
and 36 passed preflight. All 36 are frozen. The published trace's successful
effort label is not an outcome or feature. Fresh paired executions supply every
reward used by this experiment.

The frozen language mix is 16 Bash, 8 Python, 6 C, 2 C++, 2 TypeScript, 1
assembly, and 1 `ts`. Selection seed is `20260730`.

## Target seal

Only label-free DeepSWE identities and normalized prompts were compared during
source preparation. The frozen source has zero exact task-ID overlap and zero
normalized-prompt overlap with the 113 DeepSWE tasks. DeepSWE outcomes,
per-effort scores, costs, and embeddings remain unopened during source
execution, fitting, selection, and promotion.

## Execution matrix

- Provider model: `gpt-5.6-luna`
- Arms: `low`, `medium`, `high`, `xhigh`, `max`
- Independent attempts: `0`, `1`, `2`
- Tasks: 36
- Full cells: 540
- Reward: `1.0` only when the deterministic verifier passes, protected files
  remain intact, the agent process succeeds, and the requested model is
  attested; otherwise `0.0`
- Task-arm reward for fitting: mean of the three independent attempts
- Task-arm cost for fitting: mean of the three independent attempts
- Retry policy: retry infrastructure or transport failures only. Never retry a
  valid model failure or verifier failure.
- Cost provenance: provider-reported cost when available, otherwise the
  versioned trace-token estimate. Missing exact metering is not a launch gate.
- Authorized experiment ceiling: USD 20,000 total. The user monitors provider
  billing externally.

Attempt 0 may reuse pre-existing cells from the earlier v17 and v19 Moonshiner
matrices when, and only when, task identity, seed fingerprint, Moonshiner
commit, requested model, requested effort, attested observed model, verifier
integrity, and raw-trace hash all validate. The 36-task cohort was selected
without reading those outcomes. Thirty-one cohort tasks have five reusable
effort cells each. The other five attempt-0 tasks and all attempt-1 and
attempt-2 cells are fresh executions. Reused cells retain their original cost
and usage provenance and are not billed again.

Raw traces and managed workspaces remain on experiment-owned E2B sandboxes
during execution. Compact outcomes, hashes, and manifests are synchronized
before those exact sandboxes are terminated. No foundation-model weights are
downloaded or persisted.

## One paid smoke

Run exactly four cells before the full matrix:

- `bash-it-filesystemcheck` at `high`
- `bash-it-filesystemcheck` at `max`
- `behavior-dependency-planning-0142` at `high`
- `behavior-dependency-planning-0142` at `max`

The smoke passes only if all four cells persist, model attestation is exact,
trace-derived usage is present, trace hashes match, protected files remain
intact, workspaces are removed, and a fresh worker resumes without duplicating
a persisted cell. Smoke rewards do not influence candidate selection.

## Grouping and controls

Families are grouped before cross-validation. `bash-it-*`,
`behavior-dependency-planning-*`, and `vcf91-*` are explicit families. Other
tasks group by their frozen category. No family may occur in both train and
test for an outer prediction.

Selection uses nested grouped cross-validation. Candidate family and traffic
fraction are chosen only inside each outer training fold. The outer heldout
predictions are pooled once for the promotion test. Required controls are:

- best static effort on each outer training fold;
- a matched task-blind randomized `xhigh`/`max` mixture;
- shuffled uplift labels;
- a constant-score router;
- monotonicity and per-family stability reports.

## External promotion

The source experiment promotes exactly one route rule only if all conditions
hold on nested outer heldout predictions:

1. Predicted `max` uplift has positive Spearman correlation with observed
   `max - xhigh` uplift.
2. The selected route has positive reward advantage over its matched task-blind
   mixture.
3. The family-cluster bootstrap 95% lower bound for that advantage is above
   zero.
4. The routed point is not dominated by the best static effort in both reward
   and cost.
5. The shuffled-label router does not satisfy the same gate.
6. Every cell is gradeable, all three attempts are present, and all target-leak
   flags remain false.

If the gate fails, report the negative result and do not open DeepSWE. If it
passes, freeze the final lightweight route artifact, record local inference
latency, and run exactly one untouched DeepSWE v1.1 transfer against the best
static and matched task-blind controls. No DeepSWE refit, threshold tuning, or
second transfer is permitted.

## Smoke result

The one paid smoke ran from commit `dd068b29`. All four cells passed with exact
`gpt-5.6-luna` attestation, protected-file integrity, trace-derived usage, and
hash-verified raw traces. The second worker invocation reported four persisted
cells and zero pending cells, and the outcomes hash remained
`399f5bc57c01cbff3d9c7bbf36c664f965b863e30ba96e1c01328625a68b102d`.
The trace-derived cost estimate was USD 0.0329324. E2B sandbox
`ikhhw5dszmnp81xzuuc86` was terminated after artifact synchronization. A local
verifier initially joined the attempt directory incorrectly; verification was
corrected against the downloaded archive without rerunning a provider cell.

## Source result

The full source matrix completed with all 540 preregistered cells. Attempt 0
contains 155 audited pre-existing cells and 25 fresh cells. Attempts 1 and 2
contain 180 fresh cells each. The 385 new cells cost USD 6.1687718 by the
trace-derived estimate. The 155 reused cells retain USD 4.4673053 of original
cost provenance and caused no new provider calls. All three experiment-owned
matrix sandboxes were terminated after raw-trace synchronization and
verification. The attempt-0 reservation guard stopped after five persisted
cells because its local sub-cap was too tight; the retained sandbox resumed the
remaining 20 cells without duplication after the sub-cap was corrected.

The permissive grouped out-of-fold screen failed before nested selection was
warranted:

- `low`: 96.30% mean pass rate, USD 0.5005 mean-attempt corpus spend
- `medium`: 83.33%, USD 0.5406
- `high`: 75.93%, USD 0.8527
- `xhigh`: 78.70%, USD 0.8419
- `max`: 78.70%, USD 0.8097

`max - xhigh` was positive on 3 tasks, negative on 5, and equal on 28. The best
real scorer had uplift Spearman `-0.0469`, zero advantage over the matched
task-blind mixture, and family-bootstrap 95% interval
`[-0.00392, 0.00376]`. A shuffled-label scorer was better than every real
scorer, with Spearman `0.1567` and apparent advantage `0.01852`. Twenty of 36
tasks violated monotonic effort ordering. `low` strictly dominated the routed
region on both reward and cost.

The exploratory report SHA-256 is
`07888aa79a05518bcdf061378d3fe41032e2091b2ddeded331f1b8600ed7a90b`.
Because even this optimistic screen fails all three primary external gates, the
stricter nested selector cannot promote a candidate. DeepSWE remains sealed and
no target transfer is authorized from this corpus.
