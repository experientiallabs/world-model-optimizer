# Token compression: method tiers and findings (living doc, the track's canonical record)

Track brief: the serving path gains a compression stage (request -> compress -> route ->
provider call) and the policy artifact becomes JOINT (per task-cluster: compression config
+ model). This doc mirrors routing-method-tiers.md: every method tried lands here with its
verdict and evidence pointers; the PR carrying it is the track's living research PR.

## Binding methodology (violations invalidate a result)

- Accounting: every savings claim is CACHE-ADJUSTED effective cost per completed task,
  compressor inference cost and latency included. Prefix-stability (deterministic,
  append-only under conversation growth) or scoped-outside-the-cached-prefix is a hard
  requirement: breaking the provider prompt cache trades a ~0.9x discount on the prefix
  for the tokens saved.
- Evaluation: closed-loop on held-out wm scenarios (generate/execute/verify), 5 split
  seeds, PAIRED-BY-SEED vs the uncompressed baseline; power rule = 3+ seeds AND 30+ test
  scenarios else "candidate"; judge-noise discount 15-17% on wm corpora; the real-benchmark
  confirmation leg (tau-bench-real) for headline claims.
- Controls, mandatory in every accuracy grid: random token removal at matched ratio and
  truncation at matched ratio. A compressor that does not beat both has learned nothing.
- Long-tail failure hunting, mandatory: read the 10 worst per-scenario regressions per
  grid family, categorize which removed token classes changed answers (numbers, entities,
  negations, tool syntax); categories feed the per-cluster risk tiers.
- Live-run budgets: stored matrices cannot simulate compression accuracy (it changes the
  model's input); every live grid needs a master-approved cost projection first.
- Data conventions: runs as RunRecord JSONL in ~/Desktop/Projects/wmh-compression-data/
  runs/<chat>.jsonl (fit outputs OUT of params), findings per chat in findings/<chat>.md,
  cohorts never merged across capture configs (s80 vs 25-scen lesson).

## Method families under test (Silen's cut: heuristic, symbolic, learned; build AND buy)

| family | examples | prefix-stability prior | status |
| --- | --- | --- | --- |
| heuristic | self-information filtering, dedup, recency windows | deterministic by construction | round 0 audited (verdicts below) |
| symbolic | AST/syntax-aware pruning, template dedup, schema-aware tool-log compaction | deterministic by construction | round 0 audited (verdicts below) |
| learned (build) | LLMLingua-2-style token classification, cheap-LM forward-pass scorer | deterministic IF greedy/thresholded, must be tested | round 0 audited; accuracy leg next |
| hosted (buy) | The Token Company Bear-2, Compresr et al. | unknown | CUT (Silen ruling 2026-07-26: open source only; buy-vs-build closed as BUILD) |

Lit review (citation-screened) lands in wmh-compression-data/findings/lit-review.md and
gets summarized here. GPU resources: h100-dev-box-6 and h100-dev-box-3 (2x H100 each,
running), a100-backup-1 (1x A100, running); h100-dev-box is OCCUPIED (vllm work), do not
touch.

## Lit review verdict (citation-screened, 2026-07-25; full doc in the track data root)

The reframing finding: CACHING BEATS COMPRESSION on any reusable prefix (reads bill ~0.1x
on 100% of tokens; a 5x compressor bills survivors at 1.0x = 0.2x), so the track's scope
is COMPRESS WHAT IS NOT CACHEABLE. The only two end-to-end billed-cost studies disagree in
sign (one measured +6.8% cost from a 38% token reduction because it broke caching and
verbatim edit anchors; the only tau-bench study found query-agnostic + cache-control wins
while query-aware compression cost +40% for identical quality) - both are 0-cite 2026
preprints, cited as directions only, and both support the same reconciling rule above.
Prefix stability is the field-wide failure: every major method selects against a
per-input percentile, so compressed prefixes churn every turn; absolute budgets are
append-only, ratio budgets never are. The soft-prompt/latent family (Gist, AutoCompressor,
ICAE, xRAG) is dead for us on serving-contract grounds (needs model internals; cannot be
prefix-cached). The field has never run a ratio-matched random-removal control in the
LLMLingua line, never published an append-stability test, and never evaluated agentic
tool-output workloads rigorously - those three gaps are exactly rounds 0-1 of C1.

Source-strength caveat on the headline (from the review's own cross-check): the two
billed-cost studies are SPECULATIVE-tier (0 cites; one has a stated conflict of
interest), and the traffic-redundancy numbers (Preble's 85-97% prefix sharing,
TraceLab's ~19% new tokens) come from sources that are not methodologically independent
of each other, with TraceLab itself speculative. The scoping rule therefore rests on
provider PRICING ARITHMETIC (0.1x cache reads, which is not in dispute) plus directional
evidence - and Q2 (measure prefix-sharing on OUR traffic) exists precisely so no design
decision rests on those third-party redundancy figures.

Track prerequisite discovered: the tracker-side cost path (wmh/tracking/pricing.py) has
no cache tiers and cache writes are captured nowhere (the pool/serving path is correct).
C3 item 0; no savings number ships before it lands.

## Verdicts

### Round 0: the append-stability audit (C1, 2026-07-25, $0, no API calls)

The track's cheapest disqualifier, run before any live accuracy spend and, to our
knowledge, the first published-style append-stability numbers for any compression
method. Setup: 12 methods x 5 corpora (tau-bench, terminal-tasks, swe-bench,
financebench, tau-bench-real), 120 multi-turn transcripts rebuilt from the routing
matrices' stored task + replies, churn measured at every turn boundary. Churn = the
fraction of the previously emitted compressed prefix that is no longer a byte-for-byte
prefix after one appended turn (0 = append-only = provider cache survives). Code:
wmh/research/compression.py (+tests) and .agents/scripts/run_compression_audit.py;
evidence: .agents/docs/research/compression_round0/ and the track data root
(runs/c1.jsonl, findings/c1.md). All methods passed same-input-3x determinism (CPU
fp32).

| method | append-only | churn mean | token ratio | verdict (kill bar 1) |
| --- | --- | --- | --- | --- |
| head-truncate-absolute | 5/5 | 0.000 | 0.38 | survives |
| head-truncate-ratio | 5/5 | 0.000 | 0.47 | survives (see correction below) |
| dedup-keep-first | 5/5 | 0.000 | 0.93 | survives; finds little (see caveat) |
| per-turn-truncate-at-append | 5/5 | 0.000 | 0.60 | survives; strongest free structural candidate |
| json-minify | 5/5 | 0.000 | 0.98 | survives; finds almost nothing (see caveat) |
| selective-context-absolute | 5/5 | 0.000 | 0.25 | survives |
| llmlingua2-fixed-threshold | 5/5 | 0.000 | 0.60 | survives; the accuracy leg is the live question |
| selective-context-percentile | 0/5 | 0.302 | 0.58 | DEAD for cached prefixes; non-cacheable segments only |
| llmlingua2-percentile (stock rule) | 0/5 | 0.580 | 0.57 | DEAD for cached prefixes; non-cacheable segments only |
| rolling-observation-mask | 0/5 | 0.258 | 0.70 | DEAD for cached prefixes; truncate-at-append replaces it |
| tail-recency-window | 0/5 | 0.611 | 0.43 | DEAD for cached prefixes |
| random-removal (control) | 0/5 | 0.986 | 0.51 | control only |

The headline mechanism: THE SELECTION RULE DECIDES PREFIX STABILITY, THE SCORER DOES
NOT. The same scorer (GPT-2 self-information, or LLMLingua-2 177M keep-probabilities)
churns 24-81% of the emitted prefix per turn under the stock per-input percentile rule
and is byte-stable append-only under a fixed absolute threshold, on every corpus.
Stock LLMLingua-2 selection forfeits essentially the whole provider cache every turn
(worst churn 0.81 on swe-bench); its shipped-but-never-evaluated fixed-threshold path
survives at ratio 0.51-0.68.

Corrections and caveats carried forward:

- Correction to the lit review's truncation table: head-keep truncation with a RATIO
  budget IS append-only (round(ratio x n) is nondecreasing, so the kept prefix only
  extends; pinned by a test). "Ratio budgets are never append-only" is a fact about
  percentile selection, not head-keep truncation.
- The audit transcripts carry agent replies but not environment observations (the
  routing matrices do not store them), so token-reduction headroom is understated for
  every method and especially the symbolic family (dedup found 3-12%, json-minify
  0-4% on replies-only text; the literature puts observations at ~84% of agent-turn
  tokens). Stability verdicts are unaffected: stability is a property of the
  algorithm, not the text mix.
- Ratio-matching rule for all future accuracy grids: controls must match each
  method's ACHIEVED token ratio per corpus (selective-context-absolute calibrated to
  keep ~50% of line-units kept only 13-37% of tokens).
- Our scorers run per-turn/per-unit local; stock implementations chunk the whole
  prompt at global 512-token boundaries, so the percentile churn rows are lower
  bounds on the stock implementations.
- Learned-method latency (4.5-6 s/10k tokens) is local-CPU-fp32 and provisional; the
  H100 leg (realistic batch, hosting amortized) gates kill bar 3, not this round.

### Acceptance benchmark (C1's second deliverable, seam agreed, implementation next)

A cell = (compressor config, model) evaluated closed-loop on held-out wm scenarios by
`evaluate_pool(..., provider_factory=...)` (wmh/env/closed_loop.py) with a
CompressingProvider wrapper that compresses request context before the call and
records raw vs compressed tokens + compressor wall time. Cost is cache-adjusted via
`PoolEntry.cost_usd`; matched-ratio random-removal and truncation controls are just
two more compressor configs, so every grid carries them by construction; output is
RunRecord JSONL so the routing dashboard's power rule and paired-by-seed conventions
apply unchanged, and C2 runs its joint grids through the same entry point. First live
corpus: financebench-s80, after a master-approved cost projection.
