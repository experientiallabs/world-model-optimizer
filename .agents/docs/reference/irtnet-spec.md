# IrtNet implementation spec (extracted from the reference repo, 2026-07-24)

Source: ~/Desktop/Projects/router-refs/IrtNet (arXiv 2510.00844). Extracted line-by-line by a
study agent; verified citations. This is the replication contract for task #14.

## Core
2PL IRT: P(correct|m,q) = sigmoid(alpha_q . theta_m - beta_q). theta_m = learned per-model
ability embedding [num_models, 232] (the ONLY per-model params). Query: FROZEN
all-mpnet-base-v2 768d (raw, NOT L2-normalized). Head: shared Expert(768->512 ReLU
Dropout(0.5)->256) + DENSE MoE (39 experts, same shape; top_k==num_experts so gating softmax
weights ALL experts and the load-balance bias is INERT) -> h 256 -> Linear(256,1)=beta_q,
Linear(256,232)=alpha_q (alpha dim MUST equal theta dim).

## Training
BCEWithLogitsLoss on logits; rows = (model_id, prompt_id, prompt, label 0/1, category);
majority-vote labels, ties->1. Adam lr 1e-4 wd 1e-4, batch 2048, 30 epochs,
ReduceLROnPlateau(min, 0.1, patience 2) on val LOSS, best ckpt by val ACCURACY, early stop 5.
Train-only: dropout 0.5 + Gaussian embedding noise 0.05. Seed 42. Sample efficiency (<4% data)
= factorization: every query supervises ALL model thetas (1K queries x 112 models = 112K pairs).

## Routing
argmax_m (alpha_q . theta_m - beta_q); beta drops out of argmax (matters only for absolute P /
abstention / cost). Per query: 1 encoder call + 1 head forward; then one matvec Theta @ alpha.
NO cost term in paper/code - we add ours: argmax [P - lambda*cost_m] or threshold-then-cheapest.

## Serve without torch: CONFIRMED
Export from state_dict: Theta [M,232]; shared W/b pairs; gating Wg[39,768]+bg; 39 expert W/b
pairs; heads Wa[232,256]+ba, Wb[1,256]+bb. Forward = matmuls + ReLU + softmax + sigmoid
(vectorize experts by stacking). Ignore routed_moe.bias buffer + prompt_embedder. PERSIST
model_map (model name -> theta row) - not recoverable from weights.

## Published numbers (Table 1, micro routing acc, 112 models x 10 benchmarks)
IrtNet 67.4 (macro 62.0) vs Avengers-Pro 62.1 (54.3), EmbedLLM 60.2, MODEL-SAT 56.7,
RouterDC 54.9. ID correctness 72.2 full / 69.9 @1K queries. Ablation: MoE -> plain MLP drops
67.4 -> 64.0 (MoE mainly helps ranking via alpha).

## Pitfalls
Dense MoE (don't implement the bias update); alpha/theta dim tie; NO embed normalization;
disable dropout+noise at eval; ties->1; global model_map; COLD START = untrained theta rows
are garbage (new pool models need a few labeled pairs or a fallback); best-by-accuracy vs
scheduler-on-loss.

## Our adaptation decisions (deltas, documented up front)
1. Query encoder: our EmbedderSpec kinds (hashing / azure text-embedding-3-large) instead of
   mpnet - the head trains on any frozen embedding; mpnet would add sentence-transformers+torch
   to SERVING. Benchmark both.
2. Weight storage: 39-expert head ~ 21.8M params (~175MB JSON) - too big and overfit-prone at
   our per-endpoint data scale. Default to the MLP ablation variant (still 64.0 > Avengers-Pro
   62.1) with num_experts configurable; weights in an .npz SIDECAR next to policy.json (policy
   references it), not inline JSON.
3. Training behind an optional extra (torch); serving stays numpy-only.
