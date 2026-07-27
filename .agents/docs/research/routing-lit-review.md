# LLM routing: literature review (2023-2026)

Compiled 2026-07-23 by the optimization chat (endpoint pivot). Consumers: the routing
optimizer v1 built in this repo, and the deferred routing-algorithm + routing-benchmark
chat. Three research passes: eval conventions, methods design space, named-paper
resolution + open-source repo sweep. Fourth pass 2026-07-24 (drawing-board survey): citation
screening plus three families this doc had missed.

**Read the citation screen before treating anything here as established.** The first three
passes cited papers without checking uptake, which led to hill-climbing 0-to-7-citation
preprints as if they were methods. Full screen table (Semantic Scholar counts, venues,
institutions, and a SPECULATIVE tier) lives in `routing-method-tiers.md` under "Citation
screen (2026-07-24)". Counts inline below are from that pull. It also records an authorship
doubt on 2601.01330, whose "Avengers authors' own successor" framing does not match the
indexed author list.

## 1. Named-paper identity resolutions

Silen named several papers from memory (voice-transcribed). Resolutions, with confidence:

| Named | Resolution | Confidence | Notes |
| --- | --- | --- | --- |
| "ML Router" | No literal match on arXiv. Candidates: TensorOpera Router "TO-Router" (2408.12320, EMNLP 2024 Industry, BERT-based multi-LLM router), Meta-Router (2509.25535), MetaLLM (2407.10834). Also "LLM Router: Rethinking Routing with Prefill Activations" (2603.20895) | LOW | Needs Silen confirmation. Prefill-activations paper requires white-box access to routed models, so its mechanism does not transfer to an API-model pool |
| "T-Router" | TO-Router (2408.12320) or TagRouter (ACL Findings 2025, tag generator/scorer/decider, no code) | MEDIUM | "TO-Router" phonetic match; may collapse into the same paper as "ML Router" |
| "DARs" | DARS = Distribution-Aware Routing Supervision, in "From Sampled Outcomes to Capability Distributions" (2606.06924) | HIGH | Single-shot outcome labels are noisy; build supervision from multi-sample, multi-paraphrase distributions |
| "Route to roam" | R2R "Roads to Rome" (2505.21600, NeurIPS 2025): token-level SLM/LLM routing, only path-divergent tokens go to the big model. Code: thu-nics/R2R + HF weights | HIGH | Alternative: Route-to-Reason (2505.19435), lower phonetic fit |
| "mixLM" | MixLLM (2502.18482, NAACL 2025): contextual-bandit router, tag-enhanced embeddings, separate quality/cost heads + latency-constrained meta-selector | HIGH | |
| "Router R1" | Router-R1 (2506.09033, NeurIPS 2025, UIUC): the router is itself an RL-trained LLM, multi-round think/route/aggregate, cost-aware reward. Code: ulab-uiuc/Router-R1 | HIGH | Heavier paradigm; relevant only if we want multi-model aggregation, not single-shot assignment |

## 2. Design space map

### Predictive routers (per-query quality/cost predictors; our v2 family)
- Shnitzer et al. 2309.15789: the foundational framing, router = per-model binary
  correctness classifiers. Simplest thing that works; exactly the v2 outcome predictor.
- RouteLLM 2406.18665 (531 cites, ICLR 2025, Berkeley/LMSYS): strong-vs-weak router on
  Arena preferences; matrix factorization variant best; 2x+ cost cut at fixed quality.
  Repo: lm-sys/RouteLLM. **MECHANISM MISMATCH on our data (2026-07-24)**: CF/MF exists to
  impute a SPARSE matrix, and ours are 100% dense with zero missing cells (ours9 exactly
  1199x9 = 10,791 rows; each wm corpus exactly 25x9x2 = 450). There is nothing to impute,
  and a rank-k factorization of a 25x9 matrix has more parameters than observations. MF
  becomes relevant only in the multi-tenant setting where per-customer coverage IS sparse.
- Hybrid LLM 2404.14618 (ICLR 2024): predicts quality GAP; one test-time threshold
  slides the whole cost/quality tradeoff, no retraining. Soft labels from repeated
  sampling beat hard 0/1 labels under judge noise. Two ideas we adopt outright.
- RouterDC 2409.19886 (NeurIPS 2024): dual contrastive losses; fixes tie degeneracy
  (several models all good). Repo: shuhao02/RouterDC.
- EmbedLLM 2410.02223 (39 cites, ICLR 2025, Berkeley): per-model embedding learned from its
  correctness matrix; cheap model vectors + cross-benchmark forecasting. Same capacity
  caveat: it learns over 100+ models and we have 9, which is too few columns to fit a
  useful model-embedding space. The argument partly extends to r3's IRT head (a rank-1
  constrained MF fitting 9 abilities from 25 scenarios).
- GraphRouter 2410.03834 (ICLR 2025): inductive GNN; add a model = add a node with a
  few probe results. For roster churn; overkill at our data scale. Repo: ulab-uiuc.
- P2L Prompt-to-Leaderboard 2502.14855: per-prompt Bradley-Terry coefficients (1-param
  IRT); was #1 on Chatbot Arena Jan 2025. Repo: lmarena/p2l.
- UniRoute 2502.08773 (Google): unseen models at test time via model feature vectors on
  representative prompts; PROVES cluster-based routing approximates the Bayes-optimal
  rule with an excess-risk bound. Our v1 theory anchor.
- MixLLM 2502.18482: see table above. Three-head design matches our objective shape.
- Arch-Router 2506.16655 (Katanemo): routes to human-authored natural-language
  domain/action policies; 1.5B open weights on HF.
- Smoothie 2412.04692 (NeurIPS 2024, Hazy Research): LABEL-FREE quality estimation via
  weak supervision over output embeddings. De-risks thin/noisy-label clusters. Repo:
  HazyResearch/smoothie.
- IRT-Router 2506.01048 (ACL 2025): Item Response Theory, interpretable, strong
  cold-start. Repo: Mercidaiha/IRT-Router.
- CARROT 2502.03261: minimax theory result: a simple router predicting both cost and
  accuracy per query is rate-optimal. Supports lightweight-over-heavy. Repo:
  somerstep/CARROT.
- DARS 2606.06924: distribution-aware supervision (see table). Caveat on any training
  over single-shot outcome matrices, including reused RouterBench/RouterEval data.
- RadialRouter 2506.03880, xRouter 2510.08439 (Salesforce, RL tool-calling router,
  code + HF), Meta-Router 2509.25535, TO-Router 2408.12320: noted, lower priority.

### Clustering / kNN routers (our v1 family)
- The Avengers 2505.19797 (AAAI 2026): the v1 blueprint, near-verbatim: embed queries,
  k-means, score each model per cluster on a train split, route to nearest cluster's
  best model. Only real hyperparameter (#clusters) shown robust. 10 open ~7B models
  beat GPT-4.1 on 10/15 datasets.
- Avengers-Pro / "Beyond GPT-5" 2508.12631: adds one scalarization knob alpha over the
  same clustering router, traces the cost/quality frontier, Pareto-dominates GPT-5
  routing. This is v1 + our objective, already published.
- "Simple kNN Beats Complex Learned Routers" 2505.12601: tuned kNN over query
  embeddings matches/beats RouteLLM/GraphRouter-class routers. Works iff win-rates are
  locally smooth in embedding space; their locality diagnostic is worth running on our
  scenario embeddings before any v2 investment.
- Verdict: at 100s-1000s of scenarios, cluster-then-assign is near-optimal and does not
  overfit; learned per-query routers win mainly with big data, churning rosters, or
  non-smooth win-rate geometry.

### Cascades (fallback pattern, not primary)
- FrugalGPT 2305.05176 (784 cites, TMLR 2024, Stanford): canonical cascade with learned
  stop-scorer.
- AutoMix 2310.12963 (100 cites, NeurIPS 2024, CMU/Google): small model answers,
  self-verifies few-shot, a POMDP meta-verifier decides escalation. Works over black-box
  APIs.
- BEST-Route 2506.22716 (52 cites, ICML 2025, UBC/Microsoft - same group as Hybrid LLM):
  jointly picks model AND best-of-n budget; small model + n samples + selection
  substitutes for escalation. 60% cost cut, <1% drop.
- **Deferral THEORY, added 2026-07-24** - the recipes above tell you how, this line tells
  you when it works, and we were missing it entirely. "When Does Confidence-Based Cascade
  Deferral Suffice?" 2307.02764 (57 cites, NeurIPS 2023, Google Research: Jitkrittum,
  Gupta, Menon, Narasimhan et al.) characterizes the conditions under which the obvious
  confidence rule is and is not optimal; "Language Model Cascades: Token-Level Uncertainty
  and Beyond" 2404.10136 (130 cites, ICLR 2024, same group) extends it past sequence
  confidence. Our serve-side verify lever and the WS-A6 gated-escalation result currently
  ASSUME what these two papers analyze. Read before tuning that lever further.
- Cascade-Aware Training 2406.00060 (15 cites): SPECULATIVE, and needs training access
  we do not have for API models.
- Tradeoff: cascades see the cheap model's actual answer (stronger signal than the
  query alone) but pay latency and forfeit cache affinity. Keep as low-confidence
  fallback only.
- Offline feasibility on our data: `ScenarioOutcome.replies` is populated on ~100% of rows
  (10,738/10,791 on ours9; 450/450 per wm corpus) and `critique` on ~90% of wm rows, so
  the whole family simulates offline at zero API cost. Score it through
  `wmo.research.routing_runs.evaluate_call_sequences`, which takes a policy over the
  transcript so far and sums cost across calls; reading `reward`/`critique` from that
  transcript makes the run an oracle simulation rather than an achievable one.

### Reward / judge distillation (added 2026-07-24; untried)
- Zooter 2311.08692 (149 cites, 15 influential, NAACL 2024, Alibaba Qwen team: Keming Lu,
  Junyang Lin, + Tsinghua): distill an expensive judge's reward into a cheap scorer, then
  use the scorer where the judge is unaffordable. Established.
- Fit on our data is unusually direct because the SUPERVISION ALREADY EXISTS: the pinned
  Opus-4.8 judge's `critique` is stored on 395-448 of 450 rows per wm corpus, `reward` is
  the label, `replies` is the input, and reply embeddings are already cached
  (`<routing-data>/cache/routerbench-ours9-oai3l-replies.npy`). Note ours9 carries ZERO
  critiques (exact-match graded), so distillation trains on wm corpora only.
- Zooter's own framing - distill the judge, then route from the QUERY - inherits the
  query-only ceiling and is the weak use here. The valuable use is POST-HOC scoring: the
  distilled scorer is the verifier the cascade family needs and the selector best-of-n
  needs. One artifact, three families.
- Adjacent credibility: LMs (Mostly) Know What They Know 2207.05221 (1799 cites,
  Anthropic) for self-evaluation as a signal; Adaptation with Self-Evaluation to Improve
  Selective Prediction 2310.11689 (63 cites, EMNLP 2023, Google Cloud AI).

### Selective prediction / learning-to-defer (added 2026-07-24; the largest gap)
Route-vs-fallback IS learning-to-defer with the fallback as the expert. This is a
15-year-old, top-venue ML line that no earlier pass in this doc cited.
- Chow's reject option (1970): the foundational formulation.
- Selective Classification for Deep Neural Networks 1705.08500 (974 cites, NeurIPS 2017,
  Technion): risk-coverage curves, which is the right way to plot a routing guard.
- Predict Responsibly: Learning to Defer 1711.06664 (326 cites, NeurIPS 2018,
  Toronto/Vector - Madras, Pitassi, Zemel).
- Consistent Estimators for Learning to Defer to an Expert 2006.01862 (308 cites, ICML
  2020, MIT CSAIL - Mozannar & Sontag): **the naive deferral surrogate is INCONSISTENT**;
  they give a consistent one. Every guard we ship ("route if the cluster has support, else
  best-single") is an instance of the surrogate they proved inconsistent.
- Calibrated Learning to Defer with One-vs-All Classifiers 2202.03673 (86 cites, ICML
  2022, Amsterdam - Verma & Nalisnick): adds calibration on top of consistency.
- Who Should Predict? 2301.06197 (94 cites, AISTATS 2023, MIT CSAIL/IBM): exact
  algorithms, useful when the expert (our fallback) is fixed and known.
- Concrete mapping: train the deferral rule over the existing outcome matrix with
  "predict" = take the router's pick and "defer" = fall back to best-single, folding cost
  in as the asymmetric loss the L2D framework already supports. Compare against the
  current guards through `evaluate_choices`. Offline only, no spend.

### Conformal prediction / calibrated abstention (added 2026-07-24; untried)
- A Gentle Introduction to Conformal Prediction 2107.07511 (1177 cites, Berkeley) is the
  entry point; Conformal Risk Control 2208.02814 (302 cites, ICLR 2023, Angelopoulos,
  Bates, Fisch, Lei, Schuster) and Learn then Test 2110.01052 (233 cites, **Annals of
  Applied Statistics** - a statistics journal, which matters when the deliverable is a
  guarantee) are the operative results. Conformal Language Modeling 2306.10193 (168 cites,
  ICLR 2024, MIT CSAIL/Google) is the LLM-specific instance.
- Why it fits better than anything else in this doc: our one VERIFIED routing win (r1's
  kNN profile + relative threshold + statistical guard, +1.04pt paired-by-seed, 5/5 seeds)
  is a threshold hand-tuned across 7+ variants on 25-scenario corpora. Learn-then-Test
  exists to control exactly that multiple-comparisons hazard, and conformal risk control
  gives finite-sample DISTRIBUTION-FREE validity - the only guarantee class that means
  anything at n=25. It composes with r2's shrinkage and ProxRouter's tilt rather than
  competing with them.
- Cheapest item in this review: pure offline math over matrices already on disk, no API
  spend, no dependency on the judge-noise question.

### Bandits / online (v2+, after the endpoint generates logs)
- **Screen result (2026-07-24): build from the FOUNDATIONS, not from the LLM-routing
  bandit preprints.** LinUCB 1003.0146 (3422 cites, WWW 2010, Yahoo!), NeuralUCB
  1911.04462 (342 cites, ICML 2020, UCLA - note Semantic Scholar carries a duplicate stub
  record showing 17, the real record is 342), and the Thompson Sampling tutorial
  1707.02038 (1220 cites) are textbook and directly implementable over our cluster table
  as arms. Among LLM-routing-specific bandit papers only MixLLM 2502.18482 (56 cites,
  NAACL 2025) clears the screen; OptLLM 2405.15130 (20 cites, IEEE ICWS) is borderline on
  a weak venue; BARP 2510.07429 (11 cites, USC) is SPECULATIVE.
- BARP 2510.07429: multi-objective contextual bandit from bandit feedback (only the
  chosen model's outcome observed = exactly our deployed logs); one policy spans the
  tradeoff family. Attractive framing, unestablished paper - treat as a direction.
- MixLLM 2502.18482 (continual), NeuralUCB 2603.30035, dueling feedback 2510.00841
  (fits pairwise judge outputs), drifting contexts 2506.17670.
- Cold-start remedy: warm-start the bandit from the v1 cluster table.
- Real blocker is data, not method: bandits need live traffic with online feedback, and
  offline bandit evaluation on a 25-scenario matrix is not meaningful. This unlocks when
  the endpoint request log (D-SERVING-LOG / D-METERING) carries scored requests.

### Multi-turn and cache-aware (our claimed novelty)
- MTRouter 2604.23530 (ACL 2026; round-2 pass could NOT re-verify this ID, treat as
  unverified): history+model joint embeddings, MLP predicts terminal episode success;
  ACKNOWLEDGES prefix re-processing cost on switch but does NOT model it. The brief's
  v2 predictor style.
- "From Myopic Selection to Long-Horizon Awareness" 2604.12385 (2026): the ONLY paper
  that prices cross-model cache invalidation in routing (sequence-dependent invocation
  cost). Simulation-only, not wired to a real cache or serving layer. Cite as closest
  prior art; do not claim a total gap.
- Serving-layer cache literature (single model, across replicas): SGLang/RadixAttention
  2312.07104, Preble 2407.00023, CachedAttention 2403.19708, GORGO 2602.11688.
  GORGO's cost decomposition (residual prefill after prefix reuse) is the template for
  our switching-cost term.
- Novelty statement that survives review: tying real prefix-cache state to a
  CROSS-MODEL routing decision under a quality/cost/latency objective with an explicit
  switching penalty; benchmarked on multi-turn traffic.

## 3. Objective formalization
- Adopt Hybrid LLM's train-once, slide-threshold-at-test-time knob: one policy, every
  operating point, no retraining. v1 ships one balanced default position; tiers later
  are knob positions, not new machinery.
- Latency enters as a hard constraint (SLA semantics), not folded into the scalar
  (MixLLM's meta-selector pattern).
- Report the full cost/quality Pareto frontier, not a single point (RouterBench,
  Avengers-Pro convention).

## 4. Evaluation conventions (for the algorithm/benchmark chat)
- Baselines quartet: best-single-model, random, oracle (per-scenario best), all-frontier;
  plus RouterBench's Zero Router interpolation floor.
- Metrics: cost-quality Pareto frontier; AIQ (RouterBench 2403.12031, area under the
  cost-quality curve in dollars; fits the value-per-dollar framing); APGR and CPT(x%)
  (RouteLLM) when routing between two tiers. Latency: p50/p95 per scenario, no single
  canonical metric in the literature.
- Held-out: split by scenario (never by turn), ideally by CLUSTER for the OOD row;
  report in-distribution and held-out-category numbers separately.
- Precompute the model x scenario outcome matrix once (RouterBench pattern) so router
  variants compare offline on identical data. Our closed-loop eval stage produces
  exactly this matrix.
- Multi-turn honesty: single-shot eval cost overstates routing gains under prompt
  caching; our benchmark accounts for cache effects and the report states the
  assumption. No published benchmark does this.
- Reusable public assets: RouterBench 405k outcomes (withmartian/routerbench),
  RouterEval 200M+ records (2503.10657), GraphRouter's Router Dataset, RouterArena
  (2510.00202, RouteWorks/RouterArena, live leaderboard), LLMRouterBench (2601.07206,
  ynulihao/LLMRouterBench, re-implements RouterDC/Avengers/RouteLLM baselines).
  DARS caveat applies: these matrices are single-shot labels.

## 5. Failure modes (literature-flagged, mapped to us)
- Judge/label noise, the dominant one. Remedies: repeated-sampling soft labels (Hybrid
  LLM), distribution-aware supervision (DARS 2606.06924), label-free estimation
  (Smoothie) for thin clusters, pairwise/dueling feedback. Matches this repo's own
  env-reward-lottery finding; never compare across unpinned judges.
- Router overfits the eval mix; OOD drop is real (RouterDC numbers). Split by cluster.
- **Simpson's-paradox sign flips in any post-hoc/verifier feature (found here 2026-07-24,
  not from the literature).** Effort features correlate NEGATIVELY with reward when pooled
  across rows (steps -0.07..-0.29, `stop_reason == max_steps` -0.33..-0.43) because that
  is between-scenario DIFFICULTY. Within a single (scenario, model) cell the sign REVERSES
  (tau-bench steps +0.310, n_replies +0.363): at fixed difficulty the rollout that did
  more work is the better one. A selector built on the pooled sign is anti-correlated and
  loses to a coin flip. Always decompose pooled correlations into between- and within-cell
  parts before believing a feature. `wmo/research/posthoc_bounds.py` does this
  decomposition; the numbers are pinned in its test.
- Roster churn breaks static tables (motivates UniRoute model-feature vectors); our
  pool file WILL change, so policy artifacts must record the pool they were fit on.
- Tie degeneracy when several models are all good (RouterDC's motivation).

## 6. Recommendations adopted for this repo
1. v1 = Avengers-style cluster-then-assign over wm scenarios, HashingEmbedder or
   provider embeddings, k-means in numpy; soft labels via repeated sampling where
   budget allows.
2. Objective = balanced quality/cost/latency default behind a single threshold
   parameter; latency as constraint; Pareto frontier in the report artifact.
3. v2 = per-query outcome predictor (MTRouter-style, or per-model correctness
   classifiers per Shnitzer/CARROT), swappable behind the same policy artifact; run the
   2505.12601 locality diagnostic first to justify it.
4. Cache-aware switching penalty in the policy + conversation affinity in serving;
   cite 2604.12385 as closest prior art, borrow GORGO's residual-prefill cost term.
5. Cascade escalation only as a low-confidence fallback, never the primary path.
6. Bandit/online updating deferred until the endpoint generates logs; warm-start from
   the v1 cluster table.

Added 2026-07-24 (drawing-board survey; items 1-6 above stand as recorded):

7. Recommendation 5 ("cascade escalation only as a low-confidence fallback, never the
   primary path") was set before we had any post-hoc measurement, and the measurement now
   argues for promoting the family rather than keeping it as a fallback. On the SAME
   matrices, best-of-2 selection reaches operating points the query-only routers have not:
   financebench opus-4-8 0.527 @ $0.0456 vs best-single 0.478 @ $0.0636 (+4.9pt at 1.4x
   cheaper), with a free parameter-free selector. Numbers, controls and caveats in
   `routing-method-tiers.md` under "Post-hoc selection beats query-only routing on the
   ceiling"; recommendation 5 is not yet withdrawn, but it is now contested by data.
8. Ranked next steps by (credibility x expected-gain x fit): (i) cascade/deferral with a
   distilled verifier over stored replies, gated on the judge-noise decomposition;
   (ii) conformal risk control applied to r1's existing hand-tuned guard - cheapest item
   here, offline, no spend, no dependencies; (iii) learning-to-defer with a consistent
   surrogate as the principled replacement for all of our ad-hoc guards.
