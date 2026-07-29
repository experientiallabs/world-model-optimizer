# The three-corner analyses (latency-max / cost-max / quality-max)

Charter for the shared corners branch (jt/corners, one PR). Three chats work here
simultaneously; the joint-tau master coordinates and merges.

## Shared rules (binding on all three chats)

- ONE worktree (~/Desktop/Projects/wmo-corners), ONE branch, ONE PR. Shared-index
  discipline: `git pull --rebase` before every commit, stage BY PATH (never `git add -A`),
  commit fast, verify your commit landed (`git log -1 --stat`) before continuing.
- File layout: each chat owns its subdirectory here (`latency/`, `cost/`, `quality/`) plus
  shared plotting utilities in `common/` (first chat to need a helper writes it; the others
  extend, never fork). Figures follow the brand palette (AGENTS.md rule 14).
- Data sources (read-only, never regenerate): the grid matrices
  (main checkout `.wmo/jt/grid/<arm>/matrix.json` + meta), the cycle result notes and
  per-task per-arm rows (training lane run dirs), the scorecard/ladder APIs
  (wmo.optimize.scorecard), D-DIAL anchors (wmo/optimize/knn.py). Zero LLM spend in
  phase 1; charts are offline computation.
- Every chart reports ALL THREE objectives (quality, cache-adjusted effective cost per
  completed task, latency p50/p95) with provenance labels (wm_simulated vs real_episode)
  and the judge/verifier named. Compaction rungs carry "measured tradeoff, not
  recommendation" until the compaction lane's accuracy verdict lands.

## Deliverables per chat (phase 1)

1. The SHARED chart, rendered once per chat's lens: training stage (student base ->
   cycle-1 -> future gated cycles; teacher + fable-5 anchor as reference lines) on x,
   QUALITY on y, three ablation lines (distill-only / +routing / +compaction). Cycle-1's
   student is unpromoted; its holdout rows still plot, labeled "not promoted (gate
   refused: no teacher headroom)".
2. Topline graphs for the chat's own axis (latency: p50/p95 per config, latency-quality
   frontier, the corner's named mountable policy; cost: savings vs fable-5 anchor,
   effective-cost-vs-stage, the dial curve; quality: quality-vs-anchor across configs,
   the quality-max corner incl. what it pays), each annotating the other two metrics.
3. One page of findings prose per chat in its subdirectory, honest-deltas style
   (noise floor named, negative results kept).

## Phase 2 - LOCKED

Envelope-pushing research on each axis is FORBIDDEN until the full report is done and
Silen unlocks it in ~/Desktop/Projects/wmh-plan/DECISIONS.md. Do not start, scope, or
spend on phase 2.

## AMENDMENT (2026-07-27, Silen directive): ONE shared runner

Per-corner pipelines are retired. There is ONE runner, `common/build_corners.py`, owned by
the QUALITY chat (it already owns the stats and the canonical chart): it loads matrices,
fits, and episode rows ONCE, computes the full three-objective dataset ONCE (through
wmo.optimize.scorecard only), and renders per-lens figures from declarative lens specs.
Each corner's subdirectory holds ONLY its lens spec (which figures, which topline framing)
and its findings prose. The cost and latency chats refactor any standalone build scripts
onto the shared runner and delete them; a number that appears in two corners must come from
the same computation. Divergent aggregation = the two-truths bug this program kills on
sight everywhere else.

## AMENDMENT 2 (2026-07-27, Silen): quality + latency SUSPENDED; cost drives

The quality and latency chats are suspended; their directories, lens specs, findings, and
common/ contributions are FROZEN (do not delete, do not rewrite). The COST chat inherits
common/ ownership, builds common/build_corners.py (cost lens first, the frozen quality and
latency lens specs must remain renderable), and is joined by the TB2 full-pipeline cost chat
in corners/tb2-cost/. All charts continue to report all three objectives.
