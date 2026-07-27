# JiSi implementation spec (reference-repo extraction, 2026-07-24)

Source: ~/Desktop/Projects/router-refs/openJiSi (arXiv 2601.01330). Study-agent extraction with
file:line citations (run_jisi.py is the core). Replication contract for task #15.

## Architectural reality: JiSi DROPS clustering entirely
No k-means/centroids/reciprocal-rank/softmax anywhere. Training-free. Structure:
1. FIRST ROUTE: embed query (text-embedding-3-large in example config; gte-Qwen2-7B in repro),
   L2-norm, cosine vs the whole train bank; keep neighbors with sim > 0.95 * (50th-nearest sim)
   (VARIABLE count, rag_num=50 is a cap not a k); per-model profile = distance-weighted mean
   correctness over neighbors; sort.
2. SECOND ROUTE (the "mixed" part, run once): needle = top-3 models; re-rank neighbors by
   0.5*query_sim + 0.5*[0.6*response_embed_sim + 0.4*length_sim(order 2)] computed over needle
   models; keep top 50%; recompute weighted profiles on the refined subset; route = argmax.
   *** s2s mode compares the needle models' TRAIN-neighbor responses to their responses ON THE
   TEST ITEM = peeks at test responses; offline-eval only. proxy_s2s / needle_proxy_s2s are
   the DEPLOYABLE variants (train-neighbor responses compared among themselves). Any live
   claim must use proxy; label s2s numbers clearly. ***
3. ROUTE-VS-AGGREGATE SWITCH: normalize second-roll profiles by max; agg_ref_num = #models
   >= divide_t=0.8 (clamped; minus 1 if >2 refs and >13k tokens). 1 => pure routing.
4. AGGREGATION: references = top experts' response texts; aggregator = "auto" = first
   successfully-generating model walking the routed ranking; MoA synthesis prompt (optional
   with_score variant injecting profiles as confidence); strip think blocks if >90k tokens;
   single call (ppl/self-consistency path is vestigial at agg_N=1 - do not build).

## Data requirements BEYOND the correctness matrix
Per-model RESPONSE TEXTS (embeddings [N x M x D]) for train (+ test for s2s) + response token
lengths. => IrtNet runs on all our matrices NOW; JiSi phase 1 runs where replies exist:
LLMRouterBench records carry raw_output (and its example 10-model pool IS a flagship-roster
subset); our stage-B/wm matrices store replies; the frozen RouterBench adapter skipped them.

## Paper config (use main.example.json, NOT the dataclass defaults which differ)
rag_num 50, rag_thres 0.95, weighted_score true, sample_n 3, subset_p 0.5, embed_sim_mode s2s,
length order 2 coef 0.4, query_coef 0.5, re_route_mode expert_response_agg_query,
force_agg_num 3, divide_t 0.8, cut_length 13000, agg normal prompt, agg_model auto, agg_N 1,
embed truncation 7500 tokens, seed 42, split 0.7. Headline: 72.15 avg, -53% cost vs
Gemini-3-Pro; router-only 69.68 vs Avengers 68.74.

## Pitfalls (agent-verified)
Model-axis order comes from row-0 raw_output KEY ORDER (silent corruption if unstable);
response-embed caches shape-validated (rebuild on pool change); rag_num is a threshold index;
is_correct takes MAX over selected models (only honest at max_router=1); two length guards
(13k switch vs 90k think-strip); use example-json coefs (dataclass has query_coef 0.0).

## Implementation order for us
IrtNet FIRST (needs only correctness matrix + query embeddings = all matrices ready; strongest
head-to-head). Then JiSi phase 1 in PROXY mode (deployable) benchmarked on LLMRouterBench
flagship, with an s2s run only as a labeled offline-ceiling row. Phase 2 aggregation last
(needs live pool endpoints; our serving already has them).
