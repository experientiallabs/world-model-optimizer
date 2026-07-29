# tb2-cost: the full-pipeline cost corner (terminal domain)

The fourth corner chat (charter Amendment 2): not an analysis of someone else's grid, but
the PRODUCT PIPELINE run end to end under the cost-max lens, with public commands only
(`wmo build`, `wmo optimize model`, `wmo serve`) - the dry run of what a user triggers.

## Workload relabel (binding on every artifact here)

The charter named this corner "TB2". The corpus that actually feeds it is the org's
published Hugging Face bundle `experiential-labs/wmh-terminal-tasks-traces` (Silen ruling,
DECISIONS 2026-07-28): 280 real bash-agent traces / 1370 OTel spans of ONE-SHOT terminal
tasks (curl+jq registry lookups, filesystem/text processing, GitHub-via-curl, misc dev).
That is NOT Terminal-Bench 2: no TB2 task ids, no multi-step engineering episodes, and the
distill lane's measured +27pt TB2 teacher gap is not claimable against it. Every figure and
finding says "terminal-tasks", the directory keeps its charter name, and this paragraph is
the reason. Background in DECISIONS 2026-07-27/28 (TB2-cost chat entries): the TB2 episode
transcripts do not exist on this machine, the product has no harbor-to-OTel path (finding
filed), and with distillation dropped (Silen ruling, same entries) the TB2-specific
rationale no longer binds.

## Scope after the rulings

- NO DISTILLATION (Silen, 2026-07-27): no training spend, no student rung, no traffic-share
  plot. The teacher-search verdict `wmo optimize model` prints in its plan table (repo code,
  #329) is recorded as descriptive output only.
- The deliverable: routing + compaction on the terminal-tasks workload under the cost-max
  lens, mirroring the tau cohort shape (pinned pool copy, 20 test-band scenarios x 2
  episodes, arms identity / truncate / llmlingua2-endpoint, fresh compression calibration
  for terminal transcripts), with the tau-vs-terminal contrast as the findings frame.
- Figures render through `common/build_corners.py` ONLY (one-shared-runner amendment); this
  directory holds the lens spec and findings prose. Anticipated lens needs match the cost
  lens (savings-vs-anchor per rung, effective-cost per rung, dial curve) on a second
  dataset keyed by the same scorecard fields.

## Status: COMPLETE (2026-07-28) - see findings.md

Pipeline ran end to end: build ($6.80, fidelity 0.808) -> identity arm ($49.90, 494/520)
-> llmlingua2 arm ($64.14, 427/520, achieved ratio 0.777 @ dial 0.5) -> truncate arm
(partial 149/520, Bedrock outage; $30.22 incl. a written-off re-buy) -> fit/tune/report ->
figures through `common/build_corners.py --lens tb2-cost`. Total $151.06. Headline:
model selection is the whole win here (opus-5 -63% cost at +6.7pt, CI excludes zero;
routing correctly degenerates to best-single sonnet-5), and the compaction cost-inversion
reproduces. Dataset: arm matrices preserved at main checkout `.wmo/jt/tb2cost/<arm>/`.
