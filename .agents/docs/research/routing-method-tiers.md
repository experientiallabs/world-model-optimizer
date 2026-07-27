# Routing optimizer: findings + method tier list

2026-07-24, optimization chat. Canonical copy (a mirror lives in the Notion experiments area
under Research per Silen's ask). Companion docs: `routing-lit-review.md` (full literature map)
and `../proposals/routing-optimizer-v1.md` (benchmark + v1 design, approved 2026-07-24).

## Findings to date

1. **The literature supports cluster-then-assign at our data scale.** The Avengers
   (arXiv 2505.19797) is the published version of our v1 almost verbatim and shows the one real
   hyperparameter (cluster count) is robust; "Simple kNN Beats Complex Learned Routers"
   (2505.12601) shows neighborhood methods match or beat learned routers whenever win-rates are
   locally smooth in embedding space; UniRoute (2502.08773) proves cluster routing approximates
   the Bayes-optimal rule with an excess-risk bound; CARROT (2502.03261) adds a minimax argument
   for simple cost+accuracy predictors. Added router capacity at 100s-of-scenarios scale buys
   variance, not accuracy.
2. **Label noise is the dominant failure mode.** DARS (2606.06924) shows single-shot outcome
   labels are unstable; our own env-reward-lottery history (reward stdev 0.34) says the same.
   Mitigations adopted: >= 2 episodes/scenario, pinned judge (Opus 4.8), every delta reported
   with its spread, RouterBench exact-match stage as a judge-noise-free control.
3. **Cache-aware cross-model routing is a real, citable gap.** Only 2604.12385 (simulation-only,
   2026) prices cache invalidation in routing; serving-side cache work (SGLang 2312.07104,
   Preble 2407.00023, GORGO 2602.11688) is single-model. Nobody ties real prefix-cache state to
   a cross-model decision. Our serving already does default-sticky affinity; the learned
   switching rule is phase 2, gated on capturing cached_tokens.
4. **Scenario tool surface gates benchmark validity** (live finding, 2026-07-24): on bare-task
   scenarios, models that honestly decline to hallucinate tools score 0 while models that invent
   tool calls the WM plays along with score high. A matrix fitted on that teaches the router to
   prefer confident hallucinators. Fitting on agentic corpora is blocked on the wm-create tool
   surface contract; tau-bench is safe (tools.json exists).
5. **Azure retail-meter API is the pricing source of truth** (prices.azure.com): aggregator
   quotes for Kimi-K2.6 were wrong in both directions. Pool prices now pinned from published
   meters, including cache-read rates.

## The method tier list (Silen's ladder, annotated, with papers)

Ordered by when we reach for each; every step keeps the same `RoutingPolicy` artifact +
`select_model` seam so upgrades swap in without touching serving.

- **Tier 1 - NOW: clustering (Avengers replication).** Embed -> k-means -> per-cluster model
  scoring -> route to nearest cluster's model. Papers: Avengers 2505.19797 (recipe + ablations),
  Avengers-Pro 2508.12631 (the alpha cost/quality knob on top), UniRoute 2502.08773 (theory),
  kNN-beats 2505.12601 (the locality diagnostic we run to know when tier 2 is worth it).
- **Tier 2: clustering + learned predictor.** Per-model correctness classifiers
  (Shnitzer 2309.15789), matrix factorization (RouteLLM 2406.18665), contrastive query<->model
  embeddings (RouterDC 2409.19886), per-prompt Bradley-Terry (P2L 2502.14855), model embeddings
  from correctness matrices (EmbedLLM 2410.02223). Justified only by a measured oracle gap that
  tier 1 leaves unclaimed AND a failed locality diagnostic. Supervision follows DARS
  (2606.06924): multi-sample labels, not single shots.
- **Tier 3: LLM-as-router / compute-shaped routing (quality max).** Router-R1 2506.09033 and
  xRouter 2510.08439 (RL-trained router LLM, can aggregate multiple models); BEST-Route
  2506.22716 (jointly pick model AND best-of-n budget: n samples from a cheap model + selection
  can beat one frontier call on both cost and quality, which is Silen's 4-cheap-calls intuition,
  published). Costs router-side latency + tokens; only for quality-max tiers of the knob.
- **Tier 4: bandits for live updates.** BARP 2510.07429 (multi-objective, learns from
  bandit feedback = exactly deployed-endpoint logs), MixLLM 2502.18482 (continual), NeuralUCB
  2603.30035, dueling feedback 2510.00841. Warm-start from the tier-1 cluster table to skip
  cold-start regret.
- **Tier 5: cascades for escalation.** FrugalGPT 2305.05176, AutoMix 2310.12963 (POMDP
  meta-verifier over self-verification), BEST-Route again. The escalation decision sees the
  cheap model's actual answer (stronger signal than the query), at the price of latency and a
  cache-breaking model switch; use only for low-confidence clusters.
- **Tier 6: R2R + cache-aware serving.** R2R "Roads to Rome" 2505.21600 (token-level
  small/large routing; requires co-hosted models, so it activates only if the distillation leg
  gives us our own hosted model). Cache-aware switching: 2604.12385 (closest prior art),
  GORGO 2602.11688 (residual-prefill cost decomposition we adopt). Annotation: cache-aware is
  ORTHOGONAL to tiers 2-5 and starts as soon as serving captures cached_tokens; it is listed
  last only because its benchmark (multi-turn traffic replay) is the most work.

## Benchmark plan (approved)

Stage A: fit on RouterBench's public precomputed matrix (405k outcomes, 11 models; free,
offline, published baselines) until our implementation lands in the published band. Stage B:
run OUR 9-model pool over subsampled RouterBench prompts with exact-match grading (their matrix
contains their models, not ours; this stage answers "optimize directly against RouterBench with
our models" and is judge-noise-free). Stage C: wm-scenario matrices (tau-bench first;
terminal-tasks; bird-sql blocked on the tool-surface contract). Metrics: cost-quality Pareto +
AIQ (RouterBench 2403.12031), oracle / best-single-in-hindsight / random / Zero-Router
baselines, scenario-split held-out + one held-out-cluster OOD row. Budget approved ~$1-1.5k.

## The quality-vs-cost question (Silen asked back; recorded answer)

The tradeoff is the product knob, not a fixed constant. Default position = cost-saver at
quality parity (matches the "run 10x more for the same budget" claim; strongest honest
headline). The fitted policy exposes lambda so eval can sweep the full frontier
(fit-once-slide-knob, Hybrid LLM 2404.14618), and the future weighted-equation objective over
quality/cost/latency (Silen's direction) arrives as knob positions on the same artifact, with
latency as an SLA constraint rather than a scalar term (MixLLM's pattern).

## Results (2026-07-24, first benchmark round)

RouterBench 0-shot matrix (36,497 prompts x 11 models), stratified split seed 0
(25,548 fit / 10,949 held-out), fitter = faithful Avengers replication
(k=64, top_k=2, beta=6.0, seed 42). All numbers on the untouched held-out split:

| policy | accuracy | cost/call | note |
| --- | --- | --- | --- |
| best-single (gpt-4-1106, fit-chosen) | 0.7856 | $0.00327 | the bar |
| oracle (per-scenario best) | 0.9138 | $0.00024 | ceiling: +13pts AND 13x cheaper |
| random | 0.5223 | $0.00083 | floor |
| rank, hashing-512, lam=0 | 0.7882 | $0.00264 | beats best-single on BOTH axes |
| rank, hashing-1024, lam=0 | 0.7886 | $0.00253 | |
| rank, hashing-1024, lam=0.02 | 0.7723 | $0.00094 | -1.3pts for 71% cheaper |
| rank, hashing-1024, lam=0.05 | 0.7552 | $0.00069 | |
| rank, hashing-1024, lam=0.2 | 0.6938 | $0.00024 | oracle-cost territory |

Readings: (1) the pure replication clears best-single with cost playing no part in fitting -
the saving falls out of the ~25% of clusters where cheaper models genuinely rank first;
(2) the cost knob (rerank_policy, fit-once-slide) traces a real frontier and answers the
"how much cost for quality" question with data; (3) the gap to oracle (12.5pts) is the
embedder + per-query-predictor headroom - hashing trigram is lexical, and the
text-embedding-3-large comparison run is in flight. Artifacts: .wmo/evals/routerbench/.

## Implementation comparison vs the reference (ZhangYiqun018/Avengers)

Read line-by-line during implementation (core/routing/rank_router.py,
core/generate_rank_router.py); differences, all deliberate and documented in
wmo/optimize/routing.py's docstring:

| aspect | reference | ours |
| --- | --- | --- |
| embed + normalize | external gte-qwen2-7b service + sklearn Normalizer | EmbedderSpec (hashing or Azure deployment) + same Normalizer |
| clustering | sklearn KMeans k-means++/elkan/max_iter=1000/seeded | identical call |
| per-cluster score | correct/total (binary) | mean reward (graded; identical on binary) |
| ranking | accuracy desc, dict-order ties | reward desc, pool-order ties (deterministic) |
| selection | top-k softmax(-beta*dist), sum p/(rank+0.1), missing=1/999 | identical math (shared serve/eval code path) |
| artifacts | centres.npy + rankings.json + joblib normalizer | one versioned policy.json (self-describing) |
| cost | absent | absent at lam=0; stored evidence + rerank knob on top |
| routing output | top-N experts (generator may ensemble) | top-1 (single-model serving); top-N is the tier-3 ensembling step |

No accidental deltas found; the two additive extras (cluster labels for the request log,
cost evidence for the knob) do not alter lam=0 behavior.

## AIQ vs the non-predictive floor (2026-07-24)

RouterBench's own headline metric, computed with their normalization (hull area / shared max
cost) on the held-out split: **rank-router AIQ 0.7447 vs Zero-Router 0.7001** (Zero-Router =
hull of the 11 single models + their random mixes, the best label-free strategy). The fitted
hull dominates the floor across the interior cost range, e.g. at ~$0.00094/call the router
holds 0.772 accuracy where price-mixing toward gpt-4 interpolates ~0.70. Prediction is
genuinely adding value beyond price interpolation. Caveat carried from their code: AIQ's
normalization couples it to the comparison set's max cost, so we always publish hull points
alongside the scalar.

## Closing cross-check facts (2026-07-24)

- Avengers: every evaluate/* harness passes the RAW prompt into `router.route(question)`
  (grep across evaluate/*.py); no preprocessing before embedding, so the replication's
  query path matches. Paper-era artifact names pin k=64 (ranking_centers_split_k64_m22),
  config template pins top_n=2/top_k=2/beta=6.0.
- RouterBench licenses: code MIT (LICENSE carries a stray LangChain copyright line, likely a
  template slip); the HF dataset carries NO license tag. We consume it for internal
  validation and cite it; do not redistribute the data in any public artifact.

## Embedder comparison (2026-07-24): hashing ties text-embedding-3-large on RouterBench

Identical 12k-prompt subsample (seed 7), identical split, knob=0: best-single gpt-4 0.7937 @
$0.00318; hashing-1024 0.7935 @ $0.00240; azure text-embedding-3-large (3072d) 0.7930 @
$0.00262. The semantic embedder bought nothing, and its fit leg took 33 minutes against the
rate-limited deployment vs ~12s for hashing.

WHY (cluster audit of the fitted policy): hashing's 64 clusters align cleanly with dataset
identity (24 hellaswag clusters, per-subject mmlu clusters, ...) because RouterBench prompts
carry strong lexical format signatures, and dataset identity IS most of this benchmark's
routable signal (gpt-4 ranks first in 54/64 clusters; the routing win lives in the 10 clusters
led by Yi-34B/claude-v2/mixtral). LIMIT: this does not transfer to wm-scenario corpora, where
all scenarios share one format and the routable structure (if any) is semantic. Decision:
hashing stays the default for stage A/B; the comparison RERUNS per corpus in stage C before
any product-default call. Follow-up noted: compare full knob-swept frontiers, not just knob=0.

## Stage B (2026-07-24): OUR 9-model pool on certified RouterBench MCQ, exact-match

Setup: gold recovered from the matrix by consensus (23,354/26,821 certified, 87.1%, zero
parse conflicts); 1,199 stratified prompts x 9 models = 10,791 calls, $17.77, judge-free
grading; ~0.5% error rows (Azure content filters), recorded unscored. 843 fit / 356 test.

Single models on test (cost/call, accuracy): gpt-5.4-mini $0.00015/0.827; haiku-4-5
$0.00023/0.888; deepseek-v4-pro $0.00033/0.836; sonnet-5 $0.00084/0.941; opus-4-8
$0.00131/0.949; glm-5.2 $0.00242/0.812; kimi-k2.6 $0.00248/0.805; fable-5 $0.0035/0.904;
gpt-5.5 $0.00351/0.955. Oracle 0.9916 @ $0.00019.

HONEST HEADLINE: on these saturated 2023-era MCQ benchmarks, routing does NOT beat
price-mixing with our pool. AIQ ours 0.8726-0.8802 (k swept 4-64) vs Zero-Router 0.9014;
sonnet-5 alone (0.941 @ $0.00084) anchors a singles hull the fitted frontier stays under.
Best routed point (k=32, knob=0): 0.9607 @ $0.00236 - above best-single gpt-5.5 (+0.6pt,
within the ~1.7pt noise floor of 356 scenarios) at 33% lower cost and faster (routed p50
0.8-1.9s vs gpt-5.5 p50 1.87s / mean 2.33s), but single points do not beat the hull.

WHY, and why this does not kill routing: 2026 models are near-ceiling and almost TOTALLY
ORDERED on 2023 MCQ (every model 0.81-0.955); per-cluster specialization barely exists, and
843 scenarios / 64 clusters = ~13 per cluster fits rankings on noise (the no-support-threshold
characteristic + DARS 2606.06924's warning, live). Contrast stage A: the 2023 pool had REAL
specialization (Yi-34B led whole clusters over gpt-4) and routing beat both best-single and
the floor. Routing pays where models specialize; it cannot pay where quality is totally
ordered and the data is saturated.

Consequences: (1) the fitter is validated (stage A) and the honest-benchmark machinery does
its job - on saturated corpora the improvement report should recommend a STATIC policy
(sonnet-5 at 4x under gpt-5.5's cost, or haiku at 15x under with -6.7pts) and say so;
(2) the decisive test for the product is stage C, wm scenarios from customer traces, where
specialization demonstrably exists (the bird-sql smoke: haiku 1.0 vs gpt-5.4-mini 0.0 on
identical scenarios); (3) fitter iteration queue: per-cluster support thresholds, k selection
on validation, DARS-style multi-sample labels.

## LLMRouterBench flagship track (2026-07-24): the modern-benchmark validation

Per Silen's pointer, github.com/ynulihao/LLMRouterBench (2601.07206; NOT what Avengers used -
it postdates Avengers and re-implements it as a baseline). Its performance-cost track = 13
flagship 2025 models with measured costs on hard datasets (AIME, GPQA, HLE, LiveCodeBench,
arenahard, ...). Our fitter on the shared-coverage matrix (1,560 scenarios, 70/30 split seed 0,
hashing-1024, k=64):

| point | accuracy | cost/call | PerfGain | CostSave |
| --- | --- | --- | --- | --- |
| best-single (gemini-2.5-pro) | 0.7938 | $0.04770 | - | - |
| rank, lam=0 | 0.7895 | $0.00951 | -0.5% | +80.1% |
| rank, lam=0.02 | 0.7799 | $0.00495 | -1.7% | +89.6% |
| rank, lam=0.1 | 0.7682 | $0.00232 | -3.2% | +95.1% |
| oracle | 0.9861 | $0.00242 | +24% | +95% |

AIQ ours 0.7652 vs Zero-Router 0.7506: the router BEATS the price-mixing floor here, unlike
saturated stage B - this matrix is unsaturated and specialized (qwen3-235b $0.001/0.740 vs
claude-sonnet-4 $0.020/0.544), so routing has real signal to harvest. Consistent with the
paper's own findings (top routers ~= each other; gains from coarse domain structure; some
routers fail to beat Best Single on accuracy; Avengers-Pro wins the Pareto). Oracle at 0.986
shows the tier-2 predictor headroom. Full comparison against their published Avengers rows =
next (their baseline configs/seeds), but the replication is squarely in the leading-router
band on their data.

## Avengers successors (2026-07-24 lit review round 3, citation graph + deep dive)

The Avengers lineage continued; per-paper deltas verified, code links checked:

1. **JiSi / "Beyond Gemini-3-Pro" (2601.01330) - the Avengers authors' own successor**
   (Yiqun Zhang + Shanghai AI Lab core team). Fixes three Avengers limits: query-only routing
   (adds query-response MIXED routing: semantics + problem difficulty in the embedding),
   static aggregation (support-set aggregator selection), separate route/aggregate (per-query
   route-vs-aggregate switch). Router-only head-to-head: 69.68 vs Avengers 68.74; full system
   72.15 avg beats Gemini-3-Pro 71.00 at 53% lower cost. Code: github.com/magent4aci/openJiSi.
2. **IrtNet (2510.00844) - strongest direct challenger.** IRT ability x difficulty latent
   model replaces per-cluster reciprocal-rank tables; beats Avengers-Pro head-to-head 67.4 vs
   62.1 routing accuracy (35k queries, 112 models) and needs <4% of the training data (the
   sample-efficiency answer to our thin-cluster noise). Code: github.com/JianhaoChen-nju/IrtNet.
   (Related: IRT-Router 2506.01048 adds the monotonicity constraint + cold-start warm-up.
   NOTE: "MonoRouter" from an earlier agent pass is FABRICATED - does not exist.)
3. **ProxRouter (2510.09852, CMU)** - exponential-tilt reweighting bolted onto existing
   k-means/kNN scores (w ~ p*exp(-phi/tau)); +2.8 to +8.1pp OUTLIER AUC with inliers
   preserved. Best effort-to-payoff for customer traffic drift. No code found.
4. **Federate the Router (2601.22318, same CMU group)** - federated routing from sparse
   decentralized evals, k-means variant composes with our clustering: pool per-cluster stats
   ACROSS customers to beat isolated thin-data tables. Our multi-tenant setting exactly.
5. Also: MetaRouter (2606.06178; learned per-user preference replaces the static alpha, beats
   Avengers-Pro on hypervolume/IGD), Mixture of Thoughts (2509.21164; latent top-K
   aggregation, +2.92% OOD over Avengers, code github.com/jacobfa/mot), EvoRoute (2601.02695;
   online experience base + Thompson sampling, per-step agentic), MoMA (2509.07571; trained
   MoE router + TOPSIS knob), RouteJudge/ORBIT (2606.18774; eval harness with Avengers rows,
   code github.com/LAMDA-Model-Reuse/ORBIT).

Synthesis for our roadmap: tier-1.5 = ProxRouter tilt (drop-in robustness); tier-2's concrete
form = IrtNet-style IRT ability model (sample-efficient, answers thin-cluster noise, open
code) with IRT-Router's monotonicity for cold start; tier-3's concrete form = JiSi's
route-vs-aggregate switch + MoT latent aggregation; multi-tenant lever = federated per-cluster
stats. Cache-aware multi-turn cluster routing remains whitespace nobody claimed - still ours.

## CORRECTION (2026-07-24): LLMRouterBench text-duplication leak, caught by the audit loop

The first IRT run on LLMRouterBench flagship posted +12pt over the rank router - too good, so
it went through the leak audit: zero id overlap, shuffled-label control collapsed (0.698,
no reward leak), BUT 327/468 test scenarios had task text appearing VERBATIM in the fit split
(the arenahard subsets categorize the same prompts across dataset dirs; overall 1,560 -> 809
unique scenarios, nearly half duplicated). Trigram-hashing embeddings make text dupes
near-exact retrieval keys, so the learned head profited most. Adapter now dedupes by task
text (first occurrence wins, logged; regression test), tainted runs purged.

CLEAN flagship numbers (809 scenarios, 566 fit / 243 test, zero residual overlap): best-single
gemini-2.5-pro 0.8045 @ $0.05284; rank lam=0 0.7202 @ $0.01592; IRT lam=0 0.7449 @ $0.01643
(+2.5pt over rank ~= the +-2.5pt noise floor at n=243 - directional, matching IrtNet's
published direction, not conclusive here); at lam=0.1 both ~0.68 @ $0.0024 (95% cheaper).
Honest reading: on this small deduped set, routing is a cost-saver (-6-8pt for -70-95% cost),
not an accuracy win. Dup audit on the other matrices: RouterBench classic 16/36,497 (0.04%,
tables stand), ours9 0/1,199 (clean). Earlier pre-dedupe LLMRouterBench rows in this doc are
SUPERSEDED by this section.

## Credibility correction + hill-climb structure (2026-07-24, Silen review)

Citation screen (which the earlier lit passes skipped): Avengers (Shanghai AI Lab, AAAI'26)
and ProxRouter (CMU) are the established methods; JiSi (2601.01330) and IrtNet (2510.00844)
are <5-citation 2026 preprints from unestablished groups. Corrected stance: jisi/irt/tilt are
RESEARCH DIRECTIONS whose empirical results here are the evidence, not their papers.

AMENDED 2026-07-24 by the drawing-board survey, which re-pulled every count from Semantic
Scholar (exact figures in the screen table below): ProxRouter is NOT an established method - it
has **5 citations** and is a 2025 preprint; what is established is the CMU *group*, which is a
weaker claim than this paragraph made. Corrected counts: JiSi **0**, ProxRouter **5**, IrtNet
**7** (not "<5"), Avengers **33**. So all four sit under the 20-citation bar, and the
research-direction stance above applies to ProxRouter's tilt exactly as it does to jisi/irt.

Structure going forward: three specialist chats (transfer prompts routing-common/r1/r2/r3 in
~/Downloads/wmh-plan-transfer-prompts/) hill-climb retrieval (r1), cluster (r2, keeper of the
credible Avengers/ProxRouter line), and learned-ability (r3) families against the shared
interfaces: matrices + runs + findings under the routing corpus ($WMO_ROUTING_DATA),
evaluate_choices/RunRecord as the single scorer, 5-seed spreads, margin guards, leak audits,
and look-at-the-outputs discipline (all binding via routing-common.md). The master chat
(optimization) owns interfaces, benchmark sanity, cross-chat comparison, and promotion into
wmo/optimize; next master tasks = drawing-board survey beyond these three families and an
empirical audit of the benchmarks themselves.

## Benchmark-health audit (2026-07-24, master): step-cap truncation

Judge sanity passes on read samples (grounded, specific critiques). But max_steps=8 truncates
real work: crmarena 55% truncated episodes averaging r=0.05 (one sampled sonnet-5 episode had
found the correct answer per the judge's own critique and scored 0.0 for not submitting in
time); dabstep 49%. Clean wm corpora = terminal-tasks (3%) and bird-sql (12%). Full table +
consequences in the corpus findings/master.md; scaled round re-runs
exploration corpora at max_steps 16, and stop_reason mix becomes a standard health metric.

## Benchmark health: the headroom map (2026-07-24, master audit complete)

Full table in the corpus findings/master.md. Verdict: our demonstrated wins live on
SATURATED corpora (ours9/terminal-tasks/bird-sql: oracle headroom +0.03-0.04) where
cost-at-parity was the only prize; real accuracy headroom (+0.23-0.39) is on financebench /
tau-bench / continual-learning (clean-ish) and crmarena / dabstep / tau-telecom (step-cap
contaminated). Scaled-winners round targets the first three at max_steps 16, ~80 scenarios.

## Citation screen (2026-07-24, drawing-board survey): the tier list is inverted

Every count below pulled from the Semantic Scholar graph API on 2026-07-24 (`citationCount`,
with `influentialCitationCount` in parentheses). Screen bar: >=20 citations AND a known
institution, else speculative. The finding is that the three families we are actively
hill-climbing are the LEAST established work in the review, while several untried families rest
on 100-1000+ citation top-venue work.

Established (clears the bar):

| paper | cites (infl) | venue | institution |
| --- | --- | --- | --- |
| Self-Consistency 2203.11171 | 7260 (933) | ICLR'23 | Google Brain |
| LinUCB 1003.0146 | 3422 (457) | WWW'10 | Yahoo! Research |
| Scaling Test-Time Compute 2408.03314 | 1990 (146) | arXiv'24 | Google DeepMind / Berkeley |
| LMs (Mostly) Know What They Know 2207.05221 | 1799 (159) | arXiv'22 | Anthropic |
| Thompson Sampling tutorial 1707.02038 | 1220 (63) | FnT ML'17 | Stanford / DeepMind |
| Conformal intro 2107.07511 | 1177 (129) | arXiv'21 | Berkeley |
| Selective Classification for DNNs 1705.08500 | 974 (107) | NeurIPS'17 | Technion |
| Large Language Monkeys 2407.21787 | 872 (74) | arXiv'24 | Stanford / Oxford |
| FrugalGPT 2305.05176 | 784 (71) | TMLR'24 | Stanford |
| RouteLLM 2406.18665 | 531 (85) | ICLR'25 | Berkeley / LMSYS / Anyscale |
| Mixture-of-Agents 2406.04692 | 457 (58) | ICLR'25 | Together AI / Duke |
| Hybrid LLM 2404.14618 | 365 (46) | ICLR'24 | UBC / Microsoft Research |
| NeuralUCB 1911.04462 | 342 | ICML'20 | UCLA |
| Predict Responsibly (L2D) 1711.06664 | 326 (41) | NeurIPS'18 | Toronto / Vector |
| Consistent Estimators for L2D 2006.01862 | 308 (62) | ICML'20 | MIT CSAIL |
| Conformal Risk Control 2208.02814 | 302 (54) | ICLR'23 | Berkeley / Stanford |
| tinyBenchmarks 2402.14992 | 274 (31) | ICML'24 | Michigan / IBM |
| Learn then Test 2110.01052 | 233 (54) | Ann. Appl. Stat.'21 | Berkeley / Stanford |
| RouterBench 2403.12031 | 199 (34) | arXiv'24 | Martian |
| LLM Routing w/ Benchmark Datasets 2309.15789 | 171 (11) | arXiv'23 | MIT-IBM Watson |
| Conformal Language Modeling 2306.10193 | 168 (11) | ICLR'24 | MIT CSAIL / Google |
| Zooter 2311.08692 | 149 (15) | NAACL'24 | Alibaba Qwen / Tsinghua |
| LM Cascades: token-level uncertainty 2404.10136 | 130 (9) | ICLR'24 | Google Research |
| AutoMix 2310.12963 | 100 (9) | NeurIPS'24 | CMU / Google |
| Who Should Predict? 2301.06197 | 94 (9) | AISTATS'23 | MIT CSAIL / IBM |
| RouterDC 2409.19886 | 91 (15) | NeurIPS'24 | HKUST |
| Calibrated L2D one-vs-all 2202.03673 | 86 (16) | ICML'22 | Amsterdam |
| UniRoute 2502.08773 | 76 (5) | arXiv'25 | Google Research |
| Selective Prediction via Self-Eval 2310.11689 | 63 | EMNLP'23 | Google Cloud AI |
| When Does Confidence-Based Cascade Deferral Suffice? 2307.02764 | 57 (4) | NeurIPS'23 | Google Research |
| MixLLM 2502.18482 | 56 (6) | NAACL'25 | NEC Labs / ASU |
| BEST-Route 2506.22716 | 52 (4) | ICML'25 | UBC / Microsoft |
| Fusing Complementary Expertise 2310.01542 | 50 (4) | ICLR'24 | Michigan / SambaNova |
| EmbedLLM 2410.02223 | 39 (10) | ICLR'25 | Berkeley |
| Avengers 2505.19797 | 33 (4) | arXiv'25 (AAAI'26) | Shanghai AI Lab |
| Avengers-Pro 2508.12631 | 27 (2) | DAI'25 | Shanghai AI Lab |
| CARROT 2502.03261 | 25 (11) | arXiv'25 | Michigan / IBM |
| OptLLM 2405.15130 | 20 (1) | IEEE ICWS'24 | Newcastle AU (weak venue, borderline) |

SPECULATIVE (<20 citations or unestablished group) - usable as research directions, never
citable as established method:

| paper | cites | note |
| --- | --- | --- |
| Cascade-Aware Training 2406.00060 | 15 | Google, but little uptake |
| "Simple kNN Beats Complex Learned Routers" 2505.12601 | 12 | single author (Yang Li), no affiliation given - and this is what our locality diagnostic leans on, the weakest support in the tier list above |
| BARP 2510.07429 | 11 | USC |
| IrtNet 2510.00844 | 7 | our r3 line |
| ProxRouter 2510.09852 | 5 | CMU group established, paper is not |
| JiSi / "Beyond Gemini-3-Pro" 2601.01330 | **0** | see attribution doubt below |

**JiSi attribution doubt.** The Avengers-successors section above credits 2601.01330 to "the
Avengers authors' own successor (Yiqun Zhang + Shanghai AI Lab core team)". Semantic Scholar's
record for that arXiv ID lists the author order as **Shengji Tang, Weihao Lin, Jingqi Ye, Hao
Li** - Yiqun Zhang is not the first author, and the "same team's successor" framing does not
follow from the record. Anyone relying on that lineage claim should re-verify it against the PDF
before repeating it; the empirical r1 results stand on their own either way.

## Missing families (2026-07-24, drawing-board survey): what the tier list never listed

Three families absent from tiers 1-6 and from routing-lit-review.md. See that doc's new
sections 2.x for the per-family screen data and the concrete mapping onto our matrices.

1. **Selective prediction / learning-to-defer** - the largest reading gap. Route-vs-fallback IS
   learning-to-defer with the fallback as the expert, and Mozannar & Sontag (308 cites, ICML'20)
   proved the naive deferral surrogate INCONSISTENT while giving a consistent one; Verma &
   Nalisnick (86, ICML'22) add calibration. Every guard in our stack ("route if the cluster has
   support, else best-single", r1's tuned statistical guard) is an instance of the surrogate they
   proved inconsistent. Offline only, no spend. Assigned to r2.
2. **Conformal prediction / calibrated abstention** - r1's verified promote-candidate is a
   threshold hand-tuned across 7+ variants (statz0/statz05/statz1/thres90/thres98/floor40/
   floor50 in the corpus findings/r1_debug/) on 25-scenario corpora. That is precisely the
   multiple-comparisons setting Learn-then-Test (233 cites, Annals of Applied Statistics) exists
   to control, and conformal risk control (302, ICLR'23) gives finite-sample distribution-free
   validity - which is what we need at n=25. Composes with r2's shrinkage and the tilt rather
   than competing. Assigned to r1.
3. **Cascade deferral THEORY (not just the recipes)** - the tier-5 entry cites FrugalGPT and
   AutoMix but not the Google Research line that characterizes WHEN confidence-based deferral is
   optimal: 2307.02764 (57, NeurIPS'23) and 2404.10136 (130, ICLR'24). Our serve-side verify
   lever currently assumes what these papers analyze.

Also correcting a mechanism claim in tier 2: **matrix factorization does not fit our data.**
CF/MF exists to impute a SPARSE matrix; ours are 100% dense with zero missing cells (ours9 is
exactly 1199x9 = 10,791 rows; each wm corpus exactly 25x9x2 = 450). EmbedLLM-style model
embeddings need many models to learn from (it uses 100+; IrtNet used 112) and we have 9. The
same capacity argument partly applies to r3's IRT, which is a rank-1 constrained MF fitting 9
abilities from 25 scenarios - hence r3's capacity-vs-data question. MF becomes relevant only in
the multi-tenant setting, where per-customer coverage IS sparse (the federated entry).

## Post-hoc selection beats query-only routing on the ceiling (2026-07-24, master)

Reproduce with `.agents/scripts/audit_posthoc_bounds.py`; numbers pinned by
`wmo/research/posthoc_bounds_test.py`. Every router family so far predicts a model from the
QUERY. Measuring the other axis on the SAME matrices, using the 2 episodes each wm cell already
holds:

- **Ceiling.** Best-of-2 with a perfect verifier beats best-single on BOTH axes on 4 of 6 wm
  corpora - tau-bench kimi-k2.6 0.608 @ $0.049 vs best-single fable-5 0.537 @ $0.174 (+7.1pt,
  3.6x cheaper); terminal-tasks gpt-5.4-mini parity at 10.7x cheaper. Larger than anything the
  three query-only families have produced.
- **Achievable, free.** A parameter-free selector - prefer the episode that FINISHED, then the
  one that did MORE work - harvests +12%..+53% of that ceiling at 60-79% correct on decisive
  cells (pooled 67.5% over 440, z=+7.34). Best operating points: financebench opus-4-8 0.527 @
  $0.0456 vs best-single 0.478 @ $0.0636 (**+4.9pt at 1.4x cheaper**); tau-bench kimi-k2.6 0.544
  @ $0.0487 (parity, 3.6x cheaper). No extra call, no added latency, so best-of-n does not need
  an LLM verifier for a first cut.
- **The trap.** The POOLED correlation of effort with reward is NEGATIVE (steps -0.07..-0.29,
  stop_reason==max_steps -0.33..-0.43), so the obvious selector prefers FEWER steps - and that
  one is WORSE than a coin flip (35-61% correct, harvest -32%..+14%). The pooled signal is
  BETWEEN-cell difficulty; within a cell the effort sign FLIPS (tau-bench steps +0.310,
  n_replies +0.363). Anyone building a verifier must decompose before choosing a sign.
- **Controls.** Both components survive: the finish term at 69.7% on 165 decisive cells
  (z=+5.06), and - critically - the effort term at 66.2% on 275 cells where BOTH episodes
  finished (z=+5.37), which rules out max_steps truncation as the whole explanation. The finish
  term should weaken at max_steps 16; the effort term is cap-independent. Caveat: the selector
  was picked after trying ~7, so treat exact harvest percentages as directional.
  (CORRECTED 2026-07-24: these three z values were first published as +7.8 / +5.5 / +5.7, from an
  ad-hoc script that used the observed-proportion standard error sqrt(p(1-p)/n). The test is
  against a fixed p=0.5, so the null sqrt(0.25/n) is the correct denominator; the committed
  `pooled_correct_z` uses it and its values are pinned by the test. Counts and percentages were
  never affected. This is also the real source of the +5.4-vs-+5.7 gap in master's independent
  reproduction - a different SE convention, not rounding.)
- **Open gate.** The ceiling capitalizes on episode reward variance (66% of tau-bench cells
  disagree), and this harness cannot separate rollout variance from JUDGE variance. The
  judge-noise decomposition (re-judge one stored reply twice, ~450 calls/corpus) gates how much
  of the ceiling is real.

## Distilled reply verifier (2026-07-25): beats free features, still cannot pay for 2x

Reproduce with `.agents/scripts/fit_reply_verifier.py --embedder azure --seeds 0,1,2,3,4`; head code
in `wmo/research/reply_verifier.py`. Zooter path (2311.08692, 149 cites, NAACL'24): ridge over
text-embedding-3-large embeddings of the whole stored rollout transcript, trained on fit-split
rewards, numpy only. 3,977 unique reply texts embedded once and cached
(`<routing-data>/cache/wm-oai3l-replies.npz`), so reruns are free.

Selection credit on IDENTICAL reward-decisive test cells (~211/seed), pooled wm-all, 5 seeds. A
selector that cannot rank the two episodes scores 0.5, which is what makes these comparable - the
per-selector "decisive" counts in `selector_bound` are NOT (free ties on 129 cells, a continuous
score on 211, so those correct-fractions have different denominators):

| selector | credit | paired vs free | seeds better |
| --- | --- | --- | --- |
| free (finished-then-more-steps) | 0.6125 +- 0.0281 | - | - |
| **absolute (reply -> reward)** | **0.7089 +- 0.0172** | **+0.096** | **5/5** |
| pairwise (difference -> gap) | 0.6992 +- 0.0344 | +0.087 | 5/5 |
| pairwise + PCA-64 | 0.6952 +- 0.0229 | +0.083 | 5/5 |
| absolute + PCA-64 | 0.6786 +- 0.0131 | +0.066 | 5/5 |
| **shuffled-label control** | **0.5278 +- 0.0931** | -0.085 | 0/5 |

Verdict against the pre-registered bars: bar 1 (beat free) **PASSED** decisively; bar 2 (harvest
>60% of the oracle-of-2 gap) **MISSED**, absolute reaches 58%, pairwise 52%, free 31%.

Consequence for best-of-2, which is the decisive practical test: the verifier lifts fable-5's
selected-of-2 on wm-all from +0.017 over best-single (free) to +0.021..+0.036 (verifier), but the
guard demands +0.06 because 2x cost exceeds best-single's 1x. It fails on 5/5 seeds; a cost-aware
guard passes on 1/5 (opus-4-8 at 0.78x). **So bo2 still declines, and the reason is now precise:
selection is no longer the bottleneck, the 2x price is.** That points the verifier at CASCADES
rather than best-of-n - escalation pays 1x plus (escalation rate x extra) instead of 2x on
everything, so a +0.10 selection edge buys much more there.

Three things worth carrying to the next experiment:

1. **Two predictions of mine were wrong, both instructive.** The naive `absolute` head beat the
   `pairwise` head, though I expected the reverse from the between/within confound; it has ~2,891
   training rows against pairwise's ~1,126, and the semantic embedding evidently separates good
   from bad rollouts well enough globally that within-cell ranking follows. And PCA-64 HURT every
   head, so the "600 rows vs 3072 dims" capacity worry was unfounded - full-dimensional ridge at
   alpha=1 is better.
2. **Embedding choice decides whether the control is clean.** With hashing-1024 (lexical trigrams)
   the shuffled control scored 0.589, NOT chance, because trigram features encode reply length and
   length is a real within-cell quality cue. With semantic 3-large the same control collapses to
   0.528. Any future verifier must report its shuffled control, and a lexical embedding will
   flatter it.
3. Fit data provenance: these matrices predate the merged `evaluate_pool` change (error rows now
   recorded unscored rather than salvage-scored). 27 of 4,032 rows (0.67%) carry an error AND a
   reward, i.e. old-semantics salvage scores; too few to matter, recorded for completeness.

## The evaluator seam

The engineering blocker this survey flagged - `evaluate_choices` typing `choose` as
`Callable[[str], str]`, one model out, so cascades and best-of-n could not be scored through the
shared evaluator - is CLOSED as of 3a98e3b0: `evaluate_call_sequences` takes a policy over the
transcript so far and returns either the next model or `Finish(pick=i)`, summing cost across
calls. The k-th call to a model consumes its k-th stored episode, which is what makes best-of-2
simulable from a 2-episode matrix. Note its documented information boundary: transcript entries
expose `reward` and `critique`, so a policy that reads them is an ORACLE simulation (the ceiling
row), not an achievable one. The free selector above deliberately reads only `stop_reason` and
`steps`, which is why its numbers are achievable.
