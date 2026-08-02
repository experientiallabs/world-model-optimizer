# Codeforces reasoning-effort transfer protocol

Status: frozen before provider execution on 2026-07-31.

## Objective

Learn a latency-neutral policy that assigns `gpt-5.6-luna` reasoning effort
from task text and cheap structural features. Fit and select the policy only on
an external competitive-programming corpus. Evaluate on DeepSWE v1.1 exactly
once only if the external promotion gate passes.

## Frozen source

- Hugging Face dataset: `open-r1/codeforces-cots`
- Configuration: `solutions_py_decontaminated`
- Revision: `39ac85c150806230473c70ad72c31f6232fe3f41`
- Frozen task corpus SHA-256:
  `c99ac2b6637cc3c689f0c105938bc2932a40d7b3e9ed738239e10fa2b3c764c6`
- Frozen validation SHA-256:
  `b21c87f43e1172344c53d6f333c10d6f66c20695e0f4e154de5148186e62b6ce`
- Selection seed: `20260731`

The three pinned Parquet shards have SHA-256 values
`1bc738abc0ac48f25618685992b25912a28115268685cdfd27ec600edfe3c9e6`,
`db1fe217ff070106f0b00813636f8254cebce68309873004fea058ace91eb13e`,
and `20d2ef9f43099bd090798e74c1851ce588ac7a1e3fa47fd604233b97e224229c`.

Published generations are never loaded. Selection sees only problem metadata,
tests, and accepted Python reference solutions. A task is eligible only when it
is non-interactive, uses an exact-difference checker, has at least eight tests,
and its accepted reference passes the exact frozen tests inside an isolated
process. Accepted solutions validate the cohort but are absent from the saved
corpus.

The frozen cohort has 160 tasks, with 40 tasks in each of C, D, E, and F+
difficulty buckets. It spans 134 contest groups. Each task has 10 to 20 tests.

## Target seal

Only label-free DeepSWE identities and normalized prompts were compared during
source preparation. The source has zero normalized-prompt overlap with the 113
DeepSWE tasks. DeepSWE outcomes, costs, and embeddings remain unopened during
source execution, fitting, selection, and promotion.

## Execution matrix

- Provider model: `gpt-5.6-luna`
- Arms: `low`, `medium`, `high`, `xhigh`, `max`
- Independent attempts: `0`, `1`
- Tasks: 160
- Full cells: 1,600
- Generation: one Responses API call, no tools, no iterative agent loop
- Reward: fraction of frozen tests passed, from `0.0` to `1.0`
- Task-arm reward and cost: mean of the two attempts
- Retry policy: retry transport and infrastructure failures only
- Cost provenance: versioned trace-token estimate when exact provider billing
  is absent
- Authorized ceiling: USD 20,000 total, monitored externally by the user

Generated programs execute with no network, a sanitized environment, task
memory limits, task time limits, a file-size limit, and a process-count limit.
The provider credential stays in the host worker and is not exposed to the
generated program. Raw provider responses, extracted code, per-test outcomes,
usage, model attestation, hashes, and attempt identity persist before a cell is
complete. No foundation-model weights or fitted model are persisted.

## One paid smoke

Run exactly four cells on two deterministically selected frozen tasks, each at
`high` and `max`, attempt 0. The smoke passes only if all four cells persist,
model attestation is exact, trace-derived usage is present, raw response and
code hashes match, test outcomes are gradeable, and a second invocation finds
zero pending cells without changing the outcome hash.

## Grouping and controls

All cross-validation is grouped by `contest_id`, with zero contest overlap
between train and test in every fold. Nested grouped selection compares task
text and cheap structural scorers against:

- the best static effort selected inside each outer training fold;
- a matched task-blind randomized effort mixture;
- shuffled uplift labels;
- a constant scorer;
- per-bucket and per-contest stability reports.

The fitted artifact must be a small CPU policy with measured local inference
latency. It may not call an LLM at inference time.

## External promotion

Exactly one source-selected rule promotes only if all conditions hold on
nested outer heldout predictions:

1. Predicted route uplift has positive Spearman correlation with observed
   heldout uplift.
2. Reward advantage over the matched task-blind mixture is positive.
3. The contest-cluster bootstrap 95% lower bound for advantage is above zero.
4. The routed point is not dominated by the best static effort in reward and
   cost.
5. The shuffled-label control does not satisfy the same gate.
6. Every cell is gradeable, both attempts are present, and all target-leak
   flags remain false.

If the gate fails, report the negative result and keep DeepSWE sealed. If it
passes, freeze the selected lightweight policy and run exactly one untouched
DeepSWE v1.1 transfer. No target refit, threshold tuning, or second target
transfer is permitted.

## Source result

The 1,600-cell source matrix completed with exact coverage, two attempts per
task and effort, raw-response and code hashes, graded outcomes, and exact model
attestation. All five matrix sandboxes were terminated after their archives
passed local verification. The trace-token estimate was USD 72.4096456.

Static source results, using mean attempt reward and cost, were:

- `low`: 79.88% reward at USD 1.5323
- `medium`: 84.43% at USD 2.6364
- `high`: 91.47% at USD 6.7991
- `xhigh`: 85.69% at USD 10.3634
- `max`: 72.03% at USD 14.8737

The nested contest-grouped router scored 87.69% at USD 6.9710. It was 1.68
reward points worse than its matched task-blind mixture, with contest-bootstrap
95% interval `[-0.04657, 0.00846]`, uplift Spearman `-0.3723`, and static-high
dominance. The shuffled control also failed. The external gate therefore
failed and DeepSWE remained sealed. The fit report SHA-256 is
`de706785ba32195cd7bf6d49464aa436c073b2180f40170bdf18abb6c6149c76`.

The protocol exposed a source-specific execution confound. The frozen 32,768
output-token limit produced 85 incomplete max cells and 33 incomplete xhigh
cells, all with `max_output_tokens` as the reason. High had three such cells,
while medium and low had none. This result is retained as a valid negative for
the frozen protocol, but it cannot establish that task-conditioned effort is
unlearnable when higher efforts receive enough room to return a final program.
