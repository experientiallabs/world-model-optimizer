# BENCH-B: RL transfer — training agents inside a world model (DRAFT — consolidation skeleton)

> Status: DRAFT, post-audit (D91). The scoreboard is data-complete across all four benchmarks
> and survived a pre-declared rigor pass that materially revised it: val-split confirmation
> showed in-WM eval-split deltas do not replicate on uncontaminated tasks (both trained arms
> below base); kimi's GRPO "+13.7" was withdrawn after the full pinned-set rerun (paired
> −0.079); swe's "capability floor" was reworded to a format confound (the action-interface
> gap's third strike). Post-audit program answer: NO real-environment improvement beyond noise
> at 9B/smoke scale from any method; on-policy WM training is deployment-safe while offline
> SFT can be deployment-harmful (terminal −16.3, RFT2 −77.5); frontier-demo imitation dominates
> in-WM where demos exist (collusion caveat unresolved); and in-WM evaluation REQUIRES
> untouched confirmation sets — protocol iteration inflates deltas (the program's own val test
> caught it). Final tables live on PRs #73/#78 boards; this doc consolidates when they merge.

## The question (Kion, 2026-07-01)

Can closed-loop on-policy RL against a wmh world model beat off-policy SFT for improving an
agent model? ("Main lift over SFT is the CoT is closed loop from Qwen rather than imitation.")

## Headline answers (tau-bench, Qwen3.5-9B, n=40/row, GPT-5.5-era eval)

| arm | success | paired Δ vs base | one-line verdict |
|---|---:|---:|---|
| base | 55.0% | — | independently replicated (52.5%) |
| ICL-single (critique retry) | 45.0% | negative | retries truncate near-misses, not repair them |
| SDPO n=1 | 57.5% | +2.5 | specialization knife: airline +42.8, telecom −30 |
| REINFORCE++ n=4 grouped | 57.5% | +0.061 | best single-rollout-family RL row |
| SFT (LoRA) | 60.0% | +0.016 | redistributes competence; real lift ex-telecom (40→53%) |
| ICL-multi (cross-task memory) | 62.5% | +0.075 | best no-gradient row; ~$26 of context |
| **GRPO (smoke)** | **65.0%** | **+10.0 pts** | the paper-excluded arm, enabled by cheap WM rollouts |
| **SDPO n=8 (sibling demos)** | **65.0%** | ties GRPO | best mean reward; drift eliminated |

**Layer 2 — reality check (D70):** on REAL tau2, the WM-trained R++ checkpoint scores 92.5%
vs base 90.0% (paired +0.025) — WM training transfers without degradation. (Real env is far
easier than the WM eval env: 90% vs 55% base — absolute numbers are never comparable across.)

## The five findings that outlived the rows

1. **Single-rollout RL has an empty LR window here** (flat → drift → collapse across the swept
   range) — a variance problem that group-relative baselines fix; the negative *predicted* GRPO.
2. **Environment luck was the reward ceiling** (identical actions, stdev 0.34 across sessions).
   Decomposed (D62/D65): world-imagination variance (fix: temp-0 + seed_state) +
   judge-interpretation variance (fix: gold rubrics) + sampling noise (small).
3. **The full substrate fix makes the WM+judge deterministic and ground-truth-exact** (D66):
   seeded + rubric'd replays score 1.00×6, stdev 0.000 — real tau2's exact verdict. "The world
   model can replicate the real benchmark's reward signal" is a measured claim.
4. **Collapse budget ≈ signal-per-step × steps** (D63); dense per-turn rewards stabilize where
   scalar collapses (same-config A/B); rolling WM success is a live overfitting meter (D58) —
   the WM makes RL diagnosis cheap.
5. **Train-curve direction predicted every eval row** (4/4 + B3's runs) — cheap early stopping.
6. **Fidelity is a training-stability knob, not a monotone good** (B2's curve, D73-b): at fixed
   hypers, both HIGH-fidelity training envs (0.943 / 0.956) collapsed at ~ep 90 — twice,
   independently — while both LOW-fidelity envs ran healthy full epochs, and the lowest-fidelity
   backend (0.335) produced the best real-env transfer (92.5%). Mechanism: faithful envs grade
   harshly (clean-episode reward 0.54 vs ~0.70) → denser negative advantages → the D63 collapse
   budget shrinks below one epoch. "Train in the highest-fidelity sim available" is falsified
   at fixed hypers; fidelity must be matched to the optimizer (or vice versa). Product corollary:
   a WM exposes fidelity as a DIAL (down for stable training, up for harsh eval) — no real
   environment offers that.

## Cost reality

Training runs at smoke scale: $10-20 of WM steps. Eval: ~$0.18/episode (~$8-10/row).
Contrast: [real-env training economics — cite concurrency-scaling docs when merged.]

## Eval eras

- **GPT-5.5 era** (all rows above): env = GPT-5.5+RAG tau WM (pi-fidelity 0.9008), judge Opus 4.8.
- **Sonnet-5 era** (D71, pending ack): env = sonnet-5+RAG (tau fidelity 0.943, D69), judge
  Opus 4.8 + gold rubrics (D66). Era-crossing comparisons are forbidden; paired-Δ-vs-base
  within an era is the instrument.

## Pending sections (slots reserved)

- §Cross-benchmark replication (D67): gui-tasks / terminal / swe columns, both method families.
- §Fidelity→transfer curve: X = D69 cells (0.335 / 0.633 / 0.943 / 0.956) COMPLETE; B2-family Y
  partial (haiku-no-RAG 92.5% / haiku+RAG 85.0% real-env; high-fidelity points = pre-collapse
  drain ckpts, evals in flight). Honest caveat for the figure: with base at 90.0% and n=40,
  Y differences are weak — the robust curve result is finding 6 (stability), not the Y spread.
  B3-family curve pending.
- §Figures: ladder bars + the curve (brand palette, scripts/plot_trace_scaling.py conventions).

## Data provenance

Pinned artifacts: per-benchmark scenario/tools/rubric files (D26/D73/D74 — rubric tiers:
swe executable tests > tau gold actions > terminal/gui task-text), raw eval records per row
(committed .jsonl), wandb project wmh-rl-transfer, decision ledger D19-D74.
