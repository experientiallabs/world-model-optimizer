# BENCH-B2 results — SFT / PPO / REINFORCE++ vs the tau world model

*B2 arms of the BENCH-B RL-transfer ladder (D22): Qwen3.5-9B trained closed-loop against
the wmh tau-bench world model (Bedrock Haiku 4.5 backend), evaluated on the pinned
held-out scenarios against a pinned GPT-5.5-backed WM with an Opus 4.8 reward judge
(protocol D30). ICL is the coordinator's row; GRPO/SDPO are chat 3's.*

## Held-out results (20 pinned eval scenarios × 2 episodes, temp 1.0)

| arm | success rate | mean reward | episodes | notes |
|---|---|---|---|---|
| base Qwen3.5-9B | **55.0%** | 0.568 | 40/40 clean | wandb `wm_tau-eval-base-v3` |
| SFT (LoRA, 698 steps) | **60.0%** | 0.584 | 40/40 clean | `wm_tau-eval-sft-ep3-v2` |
| REINFORCE++ ckpt-0096 | **52.5%** | 0.562 | 40/40 clean | `wm_tau-eval-rpp96` |
| PPO ckpt-0096 | **54.1%** | 0.577 | 37/40 clean (3 judge-timeout errors excluded) | `wm_tau-eval-ppo96` |

95% CI on a 40-episode success rate is roughly ±15 points: **no arm separates from base
at this eval size.** The paired per-scenario view is more informative:

- **REINFORCE++ − base: mean Δ −0.005, median 0.000 (2 wins / 5 losses / 13 ties).**
  The trained policy is nearly indistinguishable from base — expected given how little
  the weights moved (86 training steps, KL-regularized LoRA, max |Δw| = 1.6e-4).
  Training-side reward on the WM did drift up within the single pass
  (quartile means 0.782 → 0.812), but it does not transfer to held-out scenarios
  at this magnitude of update.
- **PPO − base: mean Δ +0.009, median 0.000 (6 wins / 5 losses).** Same picture as
  REINFORCE++: statistically flat, tiny weight movement (75 steps, max |Δw| = 1.4e-4).
  Its training-side reward was flat-to-down within the pass (0.835 → 0.776 quartiles).
- **SFT − base: mean Δ +0.016 with heavy redistribution (5 wins / 8 losses).**
  SFT newly solves scenarios base never solves (97f8d3b7 +0.93, bf04d8b8 +0.78,
  56917256 +0.67 — tasks resembling its 97 recorded demonstrations) while breaking
  scenarios base aced (ea1f0245 −0.93, 772c0b9b −0.83). Audits show the failure mode:
  the SFT model acts without checking policy constraints (e.g. cancelling a basic-economy
  booking past the 24h window) — it imitates action patterns and, trained without think
  blocks, no longer deliberates.

## Per-domain breakdown (D35 caveat applies)

| arm | airline (14 eps) | retail (16 eps) | telecom (10 eps) | excl. telecom |
|---|---|---|---|---|
| base | 29% (r 0.31) | 50% (r 0.53) | **100%** (r 1.00) | 40% |
| SFT | **50%** (r 0.46) | 56% (r 0.57) | 80% (r 0.78) | **53%** |
| PPO ckpt-0096 | 25% (r 0.30) | 50% (r 0.56) | **100%** (r 0.98) | 39% |
| R++ ckpt-0096 | 29% (r 0.29) | 44% (r 0.54) | **100%** (r 0.99) | 37% |

**Telecom saturates (10/10) for base/PPO/R++ and is familiarity, not generalization
headroom (D35):** the 5 telecom eval tasks have ~725 near-duplicate captures in the WM's
retrieval buffer (the same traces the D32 leakage rule dropped from the RL scenario
lists), so the env simulates them with near-recorded fidelity. Identical for every arm
(still apples-to-apples), but read the non-telecom columns for signal. There, the story
sharpens: **SFT's lift concentrates in airline (29% → 50%)** — the domain with the most
surviving demonstrations — while both RL arms stay flat everywhere. SFT is also the only
arm that *drops* telecom (100% → 80%): it trained on zero telecom demonstrations and
un-learned some of the base model's saturated behavior.

## Training dynamics (wandb) — why no RL lift, empirically

Nothing in the loop is broken; the runs were **cold and short**:

- **Learning rate**: both arms inherited the algorithm groups' `lr 5e-6`. The proven IH
  GRPO recipe (adopted by the B3 arm, D36) uses **3e-5 — 6× hotter**. At 5e-6 × 75–86
  steps with grad norms ~0.3, total LoRA movement tops out at |Δw| ≈ 1.5e-4 — the flat
  eval rows are the arithmetic consequence.
- **The policy did move, slowly**: R++ `actor/kl_loss` grows monotonically
  4.8e-4 → 1.8e-2 across the run (ref-model path verified working); entropy healthy
  (0.26 → 0.31, no collapse); grad norms stable.
- **PPO's clipping and IS were identity operations**: `ppo_kl = 0`, `pg_clipfrac = 0`
  for all 75 steps, and R++ rollout-IS ratios exactly 1.0 ± 0.0 — with
  `recompute_old_log_probs=false` and low buffer age, both arms trained as plain
  on-policy policy gradient.
- **PPO's critic never converged**: `critic/vf_explained_var` stayed negative through
  all 75 steps (final quartile −0.20), so GAE advantages were mostly noise — PPO was
  effectively REINFORCE with an uncooked baseline for this run length.

**Queued follow-up (next 2-GPU window):** rerun PPO/R++ at lr 3e-5 with 2–3 epochs
(~250+ steps) and critic warmup — a config change, not a code change.

## Lift experiment (box-6): the REINFORCE++ learning-rate window is empty

The queued follow-up ran. Single-variable LR sweep on R++, everything else identical:

| lr | KL anchor | outcome |
|---|---|---|
| 5e-6 (original) | 0.01 | flat — policy barely moves (max Δw ≈ 1.6e-4) |
| 1e-5 | 0.05 | epoch 1 healthy and rising (0.64 → 0.91 within-epoch), then a **slow slide**: paired epoch-2 vs epoch-1 on the *same scenarios* −0.131 (14W/24L, n=56), entropy 0.37 → 0.21, KL accelerating — stopped at ep. ~151, early-stop checkpoint 0072 kept |
| 3e-5 (proven IH-GRPO rate) | 0.01 | **collapse**: rewards 0.81 → 0.00 by episode ~70, entropy 0.31 → 0.08, KL 2.2 by step 45; degenerate fixed point = immediate `done()` every episode |

**Reading:** the pipeline's training signal is strong — it moves the policy hard in every
regime — but n=1 binary-reward REINFORCE++ has no productive LR window on this task:
below ~1e-5 it under-drives, at 1e-5 it drifts toward reward-destroying policies after
~1 epoch, at 3e-5 it collapses outright. This is the variance problem group-relative
baselines exist to solve: the failure *predicts* GRPO (n=8 group advantages — the arm the
WM uniquely enables, running as chat 3's cell) rather than more single-rollout tuning.
The early-stopped ckpt-0072 held-out row is being measured (partial: tracking the same
flat band as the cold run). PPO at the safe profile (ratio clipping as the remaining
single-rollout stabilizer) is the last cell of this sweep.

### n=4 group-baseline cell (Silen unconstrained the n=1 setting)

`reinforce_plus_plus_baseline` (verl-native, uid-grouped), 4 rollouts/scenario in one
advantage group, lr 1e-5, KL 0.05, 2 epochs (776 episodes, 304 clean training steps,
zero failures — the most stable run of the sweep: entropy 0.23–0.37 throughout, KL
anchored ~0.02–0.03):

- **Stability solved, lift still absent on-WM**: paired epoch-2 vs epoch-1 on the
  twice-trained scenarios = **+0.002** (13W/14L, n=54). Epochs 1 and 2 both ≈ 0.80
  mean reward.
- **Why — the signal-density ceiling**: 57% of rollout groups were UNIFORM (46%
  all-success, 12% all-fail) → zero group advantage, zero gradient. The base policy
  already scores ~0.80 / ~70–80% success on the training scenarios against the WM+judge,
  so most of the compute samples from a region with no reward headroom. The ~55%
  held-out number reflects a scenario-difficulty distribution shift that on-train RL
  cannot cross by construction.
- **Held-out row (the sweep's best)**: the n=4 final checkpoint scores **57.5% / 0.629**
  (40/40 clean) — paired vs base **+0.061** (6W/4L, median 0.000). Within noise at n=40,
  but the first RL configuration to land above base on both metrics, and it is exactly
  the stable-training one. Raw records: `rpp_n4_0192.jsonl` alongside the other rows.
- **PPO-hot cell skipped with rationale**: with the signal-density ceiling identified,
  a hotter PPO would sample the same headroom-free distribution; the estimator ladder
  is closed.
- **Implication**: the binding constraint is now the *training scenario distribution*,
  not the estimator — exactly the deferred ScenarioSuite v2 work (difficulty
  calibration / harder generated scenarios, D19): train where the policy fails at
  informative rates (uniform-failure and mixed groups), not where it already succeeds.
  Scalar rewards would raise informative-group density only modestly (49% vs 43%).

Ops notes from the runs: us-east-1 Bedrock `ServiceUnavailableException` storms stalled
judge calls repeatedly — the serve script's failover chains now end in a us-west-2 link
(absorbed several flares live); the scaffold retries `new_session` connect failures with
backoff (a ~60s WM restart once burned 45 scenarios' rollout groups as instant errors).

## Training runs (97 pinned train scenarios × 1 epoch, n=1 rollout/scenario)

| arm | reward | train steps | episodes clean | WM cost (serve/judge) |
|---|---|---|---|---|
| REINFORCE++ (binary success) | mean 0.793, success 70.1% | 86 | 96/97 | $3.75 / $0.39 |
| PPO (scalar EpisodeScore.reward) | mean 0.803, success 71.1% | 75 | 97/97 | $4.52 / $0.43 |

Raw per-episode records (actions, step rewards, judge critiques, WM costs) are committed
under the agents workspace at `.agents/docs/research/wm_tau_eval_results/` (raw run
outputs stay out of `examples/` per the repo layout rules). The paired table reproduces
from those records with any per-scenario mean comparison, e.g.:

```python
import json
rows = [json.loads(l) for l in open("base_v3.jsonl") if not json.loads(l)["errors"]]
by = {}
for r in rows: by.setdefault(r["scenario_id"][:8], []).append(r["reward"])
means = {k: sum(v)/len(v) for k, v in by.items()}  # repeat per arm, subtract vs base
```

wandb project: `wmh-rl-transfer` (server runs `qwen3_5_9b_wm_tau_reinforce_real`,
`qwen3_5_9b_wm_tau_ppo_real_v2`; eval runs listed above).

## Honest reading (v1)

At this scale — single pass over ~100 scenarios, LoRA rank 32, lr 5e-6, KL-pinned —
closed-loop RL against the WM produces a policy that is statistically flat vs base on
held-out tau scenarios, and imitation (SFT) redistributes competence rather than adding
it. This is consistent with Kion's ~4% SFT experience and with the CLaaS paper's setting
being a *reference point*: detecting small deltas here needs either more training signal
(multiple epochs / more scenarios / larger LoRA lr) or a bigger eval set. The
infrastructure result stands independently: all data paths (rollout → WM reward →
TITO feedback → train → hot-reload) run end-to-end for every arm.

## Dataset facts that shape the rows

- The corpus's test split holds only **20 unique tasks** (repeat captures dominate);
  eval = all 20 × 2 episodes (D26/D30).
- The SFT dataset is **97 episodes / 698 steps**: 5 eval tasks account for 725 of 822
  train traces, and the leakage rule (never train on eval task text) drops them (D32).
- tau traces are 100% tool calls (zero message actions); judge critiques occasionally
  phrase expectations conversationally — identical bias for every arm.

## Failure analysis / infra findings (full ladder in the claas-verl journals)

See `experiments/07_02_2026_wm-tau-{sft-lora,ppo-reinforce}.md` in claas-verl and
DECISIONS D34/D40–D43: Qwen3.5 XML tool-format (vLLM `qwen3_xml` parser + scaffold text
fallback), stale-keep-alive Bedrock hangs (tcp_keepalive + same-model FallbackProvider
chains + WM warm-up probe), PPO critic token-cap and LM-head-logits OOMs (sequence
length is the lever), the wake/sleep LoRA checkpoint namespace trap (peft silently
matches zero keys; checkpoints are evaluated via direct W += α/r·B·A application), and
the non-thinking SFT template mismatch (immediate-EOS without
`chat_template_kwargs={"enable_thinking": false}`).

## Improvement sprint (v2): four arms against the flatness

Four differentiated levers, each isolated by a same-config A/B where possible. Raw
records in `.agents/docs/research/wm_tau_eval_results/{rft_sft,rpp_hard_0129,dense_0172,t0_manual060}.jsonl`;
paired Δ = mean per-episode reward delta vs base on the intersection of clean episodes.

| arm | lever | success | reward | paired Δ | verdict |
|---|---|---|---|---|---|
| A — RFT/STaR (`wm_tau-rft-sft`) | SFT on own high-reward rollouts (148 rollouts / 829 records, ≥0.9, dedup, ≤2/scenario) | 53.8% | 0.519 | −0.037 (13W/18L) | negative — did not beat demo-SFT (60.0%) |
| B — hard curriculum, ckpt 0129 (`qwen3_5_9b_wm_tau_rpp_n4_hard`) | R++ n=4 scalar on the 58 informative scenarios | 45.9% | 0.515 | −0.022 (13W/12L) | negative — run later collapsed at ~290 steps |
| C — dense per-turn rewards, ckpt 0172 (`qwen3_5_9b_wm_tau_rpp_n4_dense`) | judge `step_rewards` → per-turn credit | 42.5% | 0.547 | −0.020 (15W/15L) | eval-flat — but the **stability** result stands (below) |
| **D — temp-0 environment, ckpt manual060** (`qwen3_5_9b_wm_tau_rpp_n4_t0`) | `WMH_ENV_TEMPERATURE=0.0` training WM | **54.1%** | **0.588** | **+0.030 (15W/11L)** | only positive row; only rising train curve |

All rows n≈40 (±~15pts CI) — read direction and mechanism, not point estimates.

### Finding 1 — environment luck was the ceiling (D62)

Replaying an identical 4-step action sequence into fresh train-WM sessions scored
`{0.95, 0.15, 0.3, 0.15, 0.65, 0.95}` (stdev **0.34** on identical behavior): the env
samples at the 0.7 provider default and imagines different case circumstances per
session. The judge is *not* the noise source (stdev 0.022 across 8 rescores of one
fixed history; its temp was already 0.0). The trainer batch-whitens advantages, so this
lottery is rescaled to unit-magnitude gradient. Fix: `WMH_ENV_TEMPERATURE` in
`serve_tau_wm.py` (luck stdev 0.34 → 0.24 at temp 0; a canonical-state annex only
reached 0.21 and was shelved).

### Finding 2 — the temp-0 arm produced the bench's first rising train curve (D64)

Arm D epoch-1 thirds: reward 0.625→0.660→0.730, success 0.388→0.463→0.603 (uniform
groups 6%). Arm B on the *same seed and scenario order* was flat (0.729/0.689/0.753).
The WM-rolling-success-predicts-eval pattern (D59) held for all four arms: the three
flat-train arms evaled ≤ base; the one rising-train arm evaled above it.

### Finding 3 — the collapse budget scales with signal density (D63); dense rewards stabilize (Arm C A/B)

At fixed lr 1e-5 + KL 0.05: v1 n=4 on the diluted 97-scenario set was stable for 304
steps; Arm B (3× denser signal) collapsed at ~290 steps (KL 0.02→0.85, entropy
0.34→0.04); Arm D (denser still: 6% uniform groups) collapsed at ~260. Same config with
dense per-turn rewards (Arm C) ran all ~330 steps to completion and ended healthy
(entropy 0.34) — per-turn credit variation appears to break the uniform-hindsight-credit
dynamic. Practical rule: **drain checkpoints early and often** (Arm D's row exists only
because a manual insurance drain at step 254 caught the policy at its behavioral peak,
~10 steps before terminal collapse; behavior lags trainer metrics).

### Honest reading (v2)

The WM-as-training-env story sharpens: the harness works, and the binding constraint was
never the estimator or the LR — it is **reward channel quality**. Pinning env
temperature converts flat→learning within a run and flips the eval direction, but the
transferred effect is modest (+0.030 at n=37) because the run hits the stability budget
after ~1 epoch of productive learning. The composable next step (not run here): temp-0
env + dense rewards + per-~50-step checkpoint drains, which by Finding 3 should extend
the productive window; and B3's D59 overfitting-meter usage generalizes — the WM's
rolling train success is a reliable, cheap eval predictor across all six arms measured.

## Real-environment validation + fidelity→transfer curve (D67 program)

Full detail in PR #116; raw rows in `.agents/docs/research/real_tau_eval_results/`,
fidelity cells in `.agents/docs/research/tau_fidelity_cells/`, figure at
`docs/research/fidelity_transfer_curve.png` (regenerate:
`.agents/scripts/plot_fidelity_transfer.py`).

**Real-env validation (D70).** On REAL tau2 (real gym, real Opus user-simulator, real
grader; 20 pinned tasks × 2 trials): base **90.0%**, pinned arm (R++ n=4 ckpt-0192)
**92.5%**, paired **+0.025** (3W/2L/35T). WM-trained policy transfers with no
degradation and a slight lift against a 90% ceiling. The real env is far easier than the
WM eval env (55% vs 90% for the same base policy): WM-side and real-side absolutes never
compare; think-in-content serving (required because tau2 rejects pure-think turns)
contributes ~20pts to both rows.

**Fidelity→transfer curve.** One R++ n=4 smoke (1 epoch × 58-scenario curriculum, same
seed/order, temp-0 env, judge pinned haiku) per training-WM backend; Y = paired Δ vs base
on the REAL env; checkpoint rule uniform across points = last pre-collapse drain.

| backend | fidelity (D69) | smoke outcome | real-env Y | paired Δ |
|---|---|---|---|---|
| haiku no-RAG | 0.335 | healthy full epoch | 92.5% | +0.025 |
| haiku+RAG | 0.633 | healthy full epoch | 85.0% | −0.050 |
| sonnet-5+RAG | 0.943 | collapsed ~ep 90 (2× replicated) | 87.5% (pre-collapse drain) | −0.025 |
| opus-4.8+RAG | 0.956 | collapsed ~ep 90 | 92.5% (pre-collapse drain) | +0.025 |
| sonnet-5+RAG, collapsed ckpt | 0.943 | post-collapse | 13.8% | −0.793 |

Two claims, curve-shape only (n=1/point, n=40/row):

1. **Healthy-checkpoint transfer is flat within noise across a 3× fidelity range**
   (+0.025 / −0.050 / −0.025 / +0.025). At smoke scale against a 90% real-env ceiling,
   training-WM fidelity does not measurably change transfer — a 0.335-fidelity WM
   (haiku, no retrieval) trained as well as a 0.956 one (opus+RAG). Fidelity spend is
   not where the leverage is at this scale.
2. **Fidelity is not free for training — the fidelity/stability trade-off (D73, 3×
   replicated).** Both high-fidelity backends collapsed into a no-tool-call policy at
   ~episode 90 under the exact hypers where both haiku backends ran healthy full epochs
   (mechanism: harsher grading → denser negative advantages → the D63 collapse budget
   shrinks below one epoch). The collapsed-checkpoint row (−0.793) shows the cliff is
   total. Practical rule: when training against strong WMs, drain checkpoints early and
   select the last pre-collapse drain.

Ops finding that unblocked all of this: gpt-5.5 was lost mid-program (OpenAI account
terminated, D68) — opus-4.8 substitutes, with an Anthropic direct-API link appended to
its failover chains.

## Cross-benchmark replication: terminal-tasks (D67 leg 2)

Raw rows in `.agents/docs/research/real_terminal_eval_results/`; harness =
`packages/environment-capture/terminal-tasks/rl/terminal_real_eval.py` (real bash-in-docker, wmh judge on Opus 4.8);
28 pinned eval scenarios × 2 trials, all rows zero error records after two live-fire
harness fixes (per-command output cap; salvage+normalize of sloppy tool-argument JSON —
WM-trained ckpts emit it, verbatim replay 400s vLLM).

Training smoke: R++ n=4 vs the terminal WM (haiku env, temp-0), 150 pinned train
scenarios. The run collapsed on the D73 trajectory (KL 0.13→1.98, entropy 0.52→0.20,
no-tool episodes emerging; onset ~step 90) — **the 4th replication, first on a non-tau
benchmark: the collapse budget is benchmark-general.**

Real-env rows (base 85.7% / 0.861):

| checkpoint | step | success | reward | paired Δ |
|---|---|---|---|---|
| drain 0015 | ~30 | 80.4% | 0.794 | −0.067 (8W/12L) |
| drain 0030 | ~60 | 42.9% | 0.465 | **−0.396 (3W/38L)** |
| manual048 | 158 | 78.6% | 0.771 | −0.090 (4W/17L) |

**Terminal transfer is negative at every checkpoint** — the opposite of tau
(flat-to-positive). The trough is diagnostic: the step-60 policy issues long runs of
EMPTY bash commands (mean 13.6 steps vs base 5.7; judge critiques read "you issued 18
empty bash commands and never wrote any code"), then partially recovers format by step
158 without recovering task quality. Mechanism hypothesis: the terminal WM tolerates
malformed/empty commands and still simulates plausible outcomes, so training never
penalizes them — a reward-channel blind spot on the *counterfactual error path* that the
D12 fidelity metric (which replays recorded actions) cannot see. The real shell is not
so forgiving. Implication for WM training: fidelity on recorded actions is not
sufficient; error-path fidelity (does the WM push back on garbage actions the way the
real env does?) is a distinct, load-bearing dimension.

## Cross-benchmark replication: swe-bench train-side (D67 leg 3, eval blocked)

R++ n=4 vs the swe WM (haiku env, temp-0, 150 pinned scenarios): **no trainable signal
at 9B smoke scale** — base reward on the WM was 0.14 over the first 60 episodes (the
policy rarely solves swe tasks even simulated), advantages were uniform-failure from the
start, and the run collapsed to a no-tool policy by ~episode 250 (stopped at 290/600).
Together with tau's saturation (~0.80 base on train) this brackets the operating band
for group-relative RL against WMs: **base success must sit in the informative middle;
swe sits at the floor, tau v1 sat at the ceiling, and the hard-curriculum/temp-0 work
was about engineering the middle.** Checkpoints banked
(wm_swe_manual_early @ step 239 + periodic drains) for real-env rows once the swe
container harness (B3) exists; WM-side rows wait on the D71 eval-env re-pin.
