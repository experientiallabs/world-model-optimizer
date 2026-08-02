# External semantic kNN model by effort router

Status: frozen before model download, embedding, or fitting on 2026-08-01. DeepSWE outcomes remain
sealed.

## Question

Can a coding-specific semantic representation let WMO's native guarded kNN policy identify the
external model by reasoning-effort complementarity that character hashes and a scalar generic
difficulty score did not recover?

The 194-task SWE-rebench V2 development matrix already established substantial pairwise oracle
headroom across 15 model by effort arms. Direct character-hash estimators and a trace-trained
difficulty rank did not meet the cost-quality gates. This sequential hypothesis changes only the
pre-call task representation and uses WMO's existing kNN fitter and decision path.

## Frozen embedding source

The only embedding model is `jinaai/jina-embeddings-v2-base-code` at immutable Hugging Face commit
`516f4baf13dec4ddddda8631e019b5737c8bc250`. Inference uses
`onnx/model_quantized.onnx`, whose expected content length is 161,895,621 bytes and expected ETag
is `cdf0fdec74ef1aa8b68d360e29d9a9eee569fea6123cf494604c7e530af27c3f`, with the
same-revision `tokenizer.json`.

The pre-call text is exactly `repository`, `language`, and the task prompt, matching the prior
external development protocol. Tokenization is truncated to 1,024 tokens. The final hidden state
is attention-mask mean pooled and L2 normalized. Embedding download and inference run only on an
ephemeral E2B sandbox. No model file, tokenizer, task embedding, kNN bank, or fitted numeric state
is returned to or persisted on the Mac.

## Frozen development search

The dense external development matrix has 194 whole-task-intersected SWE-rebench V2 tasks, 15
model by effort arms, and two attempts per cell. The strongest external static arm is `sol-max`.
The pinned guard candidates are the eight nondominated or near-frontier external static arms:
`luna-low`, `luna-medium`, `luna-high`, `luna-xhigh`, `luna-max`, `terra-xhigh`, `terra-max`, and
`sol-max`.

The frozen grid is:

- guard: the eight arms above;
- neighbor count: 8, 16, 32, or 64;
- asymmetric guard z: 0, 0.5, 1, 1.645, or 2;
- cost-quality weight `pick_lam`: 0, 0.01, 0.02, or 0.03;
- relative similarity threshold: 0.95;
- absolute novelty floor: disabled;
- minimum paired evidence: 8;
- small-sample standard-error floor: enabled.

This is 640 candidates. Evaluation uses repository-grouped five-fold cross-validation repeated at
seeds 11, 23, 37, 41, and 59, with zero repository overlap between fit and heldout folds. Each
base guard and neighbor-count policy is fit once per fold. The z and `pick_lam` variants reuse the
same WMO bank and only change WMO policy fields that do not alter fitted evidence.

The baseline is `sol-max` on the same heldout tasks. A candidate is eligible only if every seed
has at least 95 percent reward retention and at least 40 percent cost savings, aggregate matched
task-blind reward advantage is positive, and no static arm dominates its reward and cost on the
same heldout observations. Selection minimizes aggregate cost, then maximizes aggregate reward,
then uses frozen candidate order.

## Blind external confirmation

If development passes, refit the selected configuration ephemerally on all development tasks and
freeze one route for the untouched 200-task SWE-rebench V2 confirmation cohort. Also freeze a
task-blind route with identical arm traffic and 128 unique null routes formed by repository-group
permutations. Run only the arms actually selected by those frozen routes, at two attempts per task,
using the pinned official verifier and retry policy.

Promotion requires all of:

1. at least 95 percent whole-task coverage with whole-task intersection only;
2. at least 95 percent reward retention and at least 40 percent cost savings versus `sol-max`;
3. positive repository-bootstrap lower bound versus matched task-blind traffic;
4. positive repository-bootstrap lower bound versus the best frozen null route;
5. no domination by an eligible static arm;
6. route decision latency p95 below 5 ms using cached vectors and WMO's real decision path;
7. zero DeepSWE outcome access and no persisted embedding model, vectors, bank, or router weights.

If every gate passes, the frozen route receives exactly one DeepSWE v1.1 transfer. DeepSWE may not
be used for repair, hyperparameter selection, representation changes, or a rerun.

## Spend and compute

The user authorized a USD 20,000 hard ceiling and monitors provider usage externally. The account
capacity is 1,000 E2B sandboxes. Model acquisition, embedding, fitting, and analysis run on E2B or
Azure, never on the local Mac. No Modal app is used. This development stage makes no provider model
calls. Rough cumulative experiment spend before this hypothesis is USD 3,025.10805955.

## Result

The result is terminal negative on external development. No provider confirmation call was
launched and DeepSWE remained sealed.

The pinned quantized ONNX model produced 768-dimensional embeddings for all 394 development and
label-free confirmation tasks in 145.07 seconds on E2B. Its SHA-256 was
`ed45870251c9f0cf656e78aab0d37a23489066df8a222bb1c8caf8a45f2cb16d`. Every one of the
640 frozen candidates was replayed across five repository-grouped split seeds, for 620,800 native
WMO route decisions. Decision latency was 0.248 ms p50 and 0.325 ms p95, below the 5 ms gate.

No candidate passed. The closest quality-safe region used `sol-max` as guard, eight neighbors,
and z 0. Its cost-quality settings retained at least 95.05 percent quality on every seed but saved
only 9.65 to 12.56 percent. The best mean matched task-blind advantage in that region was only
0.002426 reward. Semantic task proximity therefore did not recover enough task by arm interaction
to route aggressively. The negative agrees with the character-hash and scalar-difficulty results:
the available task text supports a mostly static high-effort choice, even though the same matrix
contains large oracle arm complementarity.

The E2B sandbox `iqdxxor1nstoc8foy3gcn` was terminated. The model, tokenizer, embeddings, fitted
banks, and numeric policies were not persisted. Provider spend did not increase, so rough
cumulative experiment spend remains USD 3,025.10805955.
