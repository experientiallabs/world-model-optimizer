# common/ - shared utilities and BINDING statistical conventions for the three corner chats

The quality chat owns the statistical rigor for all three corner analyses (charter). The
conventions below are binding on every chart and every number in `latency/`, `cost/`, and
`quality/`. First chat to need a helper writes it here; the others extend, never fork.

## The three statistical conventions (binding)

1. **Paired, per scenario, never unpaired spreads.** Every delta between two configurations is
   computed per scenario over the intersection of scenarios scored on BOTH sides, then
   aggregated (`stats.paired_delta`). Scenario difficulty dominates lever effects at this
   program's sample sizes (the environment-lottery lesson: per-scenario reward stdev ~0.34),
   and pairing is what removes it. An unpaired comparison of two arms' means is banned in any
   corner chart or finding.

2. **The noise floor is +-0.015 to 0.02 mean reward at these sample sizes** (20 test-band
   scenarios x 2 episodes on the grid; 20 holdout tasks x 3 attempts on cycle rows). The
   model-refresh round measured config search bottoming out at this floor and the tau grid
   forecast carried it forward. Every delta chart draws the band (`palette` has the fill
   convention); no finding may headline a delta inside it. `stats.PairedDelta` carries
   `within_noise_floor` so a renderer cannot forget.

3. **Sim-to-real framing** (joint-tau master amendment 2, 2026-07-27, binding): per-scenario
   PAIRED SIGN AGREEMENT is the primary transfer evidence (`stats.sign_agreement`). A
   model-mean rank correlation over n=4-11 models has a null SD of roughly 0.4-0.5, so
   Spearman headlines are banned; `stats.spearman_model_means` returns a value that carries
   its own descriptive-only caveat, quote it with the caveat or not at all. `top3_overlap`
   style statistics are degenerate under a constant baseline and are not used.

## Labeling rules (binding, from the charter)

- Every chart reports ALL THREE objectives: quality, cache-adjusted effective cost per
  completed task, latency p50/p95. When a source cannot supply one (cycle-1 rows carry no
  per-arm cost), the chart says so in the annotation instead of omitting it silently.
- Every series carries a provenance label (`wm_simulated` vs `real_episode`) and names its
  judge. No computed delta, CI, or agreement statistic may cross provenance; cross-provenance
  series may share a figure only as separately-labeled panels or reference lines.
- Compaction rungs carry "measured tradeoff, not recommendation" until the compaction lane's
  accuracy verdict lands.
- Missing data is named on the figure (a "pending" footnote listing what has not landed),
  never silently dropped.

## Palette (AGENTS.md rule 14, validated)

`palette.py` is the one source. Series lines use blue `#0070f3`, purple `#7928ca`, red
`#ee0000` in that fixed order, assigned by SERIES IDENTITY (distill-only / +routing /
+compaction), never by position in a particular figure. Amber `#f5a623` and teal `#50e3c2`
fail the contrast floor for 2px lines on white (1.97:1 and 1.56:1 vs the 3:1 floor), so they
are reserved for area fills and bands that carry a direct label, never a series line. The
noise-floor band is a neutral gray fill with a direct label.

## Modules

- `stats.py` - paired per-scenario statistics: `mean_with_ci` (cluster bootstrap over
  scenarios), `paired_delta` (paired bootstrap CI + exact sign test + noise-floor flag),
  `sign_agreement`, `spearman_model_means` (descriptive only, self-caveating). Pure
  stdlib + pydantic, no viz dependency.
- `data.py` - read-only loaders for the charter's data sources: grid arm matrices
  (`.wmo/jt/grid/<arm>/matrix.json` in the MAIN checkout, landing as arms merge;
  `load_arm_snapshot` falls back to pre-retry `chunk-*.json` files with completeness
  labeled, per the master's grid-timing entry), cycle-1 per-task rows
  (`episode-rows.jsonl`, 180 rows), arm metadata. Never regenerates anything.
- `latency.py` - the two latency clocks in code form (owned by the latency chat): per-task
  model seconds (scorecard contract, compressor wall included), the #295 productive-call
  rule for per-call stats, inclusive p50/p95, first-vs-warm cold-start split, and
  `config_points` aggregating every (model x arm) on all three objectives. Viz-free.
- `ablation_chart.py` - the canonical training-stage-vs-quality chart (charter deliverable 1).
  Each corner chat renders it through its lens into its own `figures/` directory.
- `palette.py` - brand palette constants and the matplotlib style (needs the viz extra).

## Running

Analysis and rendering (matplotlib comes from the viz extra):

    uv run --extra viz python .agents/docs/research/corners/common/ablation_chart.py --help

Tests are inline (`*_test.py`) but `.agents/` is outside the root gate's testpaths, so run
them explicitly:

    uv run pytest .agents/docs/research/corners/common/ -q
