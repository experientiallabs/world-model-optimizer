# GRPO + SDPO vs the tau world model (BENCH-B3)

*Fold into `results.md` when PR #73 (SFT/PPO/R++ rows) merges — same protocol, same base row.*

## Protocol

Pinned shared eval (DECISIONS.md D30/D33): all 20 test-split scenarios
(`scenarios_eval.jsonl`) × 2 episodes = 40 episodes/checkpoint; policy temperature 1.0,
max_steps 20, max_tokens 6000; env = pinned **GPT-5.5**-backed WM; reward judge =
**Opus 4.8** (Bedrock, cross-geo waterfall — same dated model in every link); success =
episode-end judge. Judge cost reported separately from WM serve cost (D12). Policy:
Qwen3.5-9B; training env = **Haiku 4.5**-backed WM (different family from the eval WM —
the strongest circularity blunting available; caveat stated regardless).

## Rows (success rate, n=40 unless noted)

| arm | overall | airline (n=14) | retail (n=16) | telecom (n=10) | wandb |
|---|---|---|---|---|---|
| base (B2 v3) | 0.550 | 0.286 | 0.500 | 1.000 | wm_tau-eval-base-v3 |
| SFT / PPO / R++ (B2) | ~flat vs base | | | | PR #73 |
| SDPO n=1, smoke ckpt (20 scen × 1 epoch) | 0.575 | **0.714** | 0.375 | 0.700 | b3-eval-sdpo-n1-ckpt |
| **GRPO, smoke ckpt (20 scen × 1 epoch)** | **0.650** | 0.429 | 0.625 | 1.000 | b3-eval-grpo-ckpt-final3 |
| GRPO, scale ckpt @48 scen (~107 steps) | 0.525 | 0.429 | 0.312 | 1.000 | b3-eval-grpo-scale48-ckpt |
| **SDPO n=8 (sibling demos), smoke ckpt** | **0.650** | 0.500 | 0.562 | 1.000 | chain-wm_tau_sdpo_n8 |

Mean rewards track success closely (base 0.568, SDPO-n1 0.573, GRPO 0.649).

## Findings

1. **GRPO is the first arm that moves, and it moves cleanly**: +10.0 pts overall from a
   one-epoch, 20-scenario smoke — +14.3 airline, +12.5 retail, telecom held at ceiling.
   SFT/PPO/REINFORCE++ were flat at comparable scale. GRPO is exactly the method the
   CLaaS paper had to exclude (single-rollout online learning cannot form groups); the
   world model's cheap counterfactual rollouts (n=8 per scenario, ~$0.06/episode on the
   Haiku WM) are what re-enable it. This is the headline claim, and the evidence so far
   supports it at smoke scale.
2. **SDPO with WM sibling demonstrations (n=8) ties GRPO at 0.650 with the best mean
   reward of any row (0.670) — and eliminates the n=1 drift.** Per-domain vs base:
   airline +21.4, retail +6.2, telecom held at 1.000 (n=1 had dropped it to 0.700).
   Failed rollouts distilling toward a successful sibling's demonstration + critique is
   both a lift and a stabilizer; the sibling channel only exists because WM group
   rollouts are cheap. Both WM-enabled group methods now beat every classical arm.
3. **SDPO n=1 specializes with drift**: +2.5 overall masking airline +42.8, retail −12.5,
   telecom −30 (regressed from saturated). Critique-only self-distillation (the n=1 weak
   form — the teacher is always the rollout itself + judge critique) is a specialization
   knife, not a general lift: the paper's forgetting axis reproduced live. The n=8
   sibling-demonstration variant (failed rollouts distill toward a successful sibling's
   demonstration + critique — only cheap because of the WM) is queued as the fix.
4. **The telecom "cross-domain probe" probes nothing at this difficulty**: base is
   already at 1.000. It functions as a regression canary instead (SDPO-n1 tripped it;
   GRPO did not).
5. **GRPO overtrains past ~30–40 scenarios at the inherited hyperparameters — and the
   training WM called it in real time.** The 97-scenario run peaked around scenario 35
   (rolling WM success 71.5% vs the 61% no-training baseline), drifted (KL 0.2→3.7,
   grads →3.8), and was killed at scenario 53 under pre-declared criteria (rolling
   success 42.5%). Its near-peak checkpoint (48 scenarios, ~107 steps) evals at 0.525 —
   below base — with the damage concentrated in retail (0.312 vs smoke's 0.625), the
   domain with the most train scenarios (61/97 = most gradient exposure). The eval
   ordering (smoke 0.650 > base 0.550 > scale48 0.525) matches the training-WM rolling
   success ordering exactly: **the WM doubles as a live overfitting meter** — you watch
   transfer degrade before paying for an eval. Rerun prescription: lr ≤1e-5 or KL coef
   ≥0.05, checkpoint every ~10 scenarios, buffer sized to avoid uid-group ring-splits.
6. **In-run training signal predicted the eval result**: the GRPO smoke's second-half
   per-scenario reward deltas (+0.067) exceeded first-half (+0.029) vs a no-training
   identical-seed run, with the biggest gains exactly on high-variance rollout groups
   (e.g. the audited exchange scenario: half-failing → 0.975). Group advantage std held
   0.28–0.30 with mean≈0; KL ≈ 9e-5.

## Interim Opus-env re-eval + the arm pin (D65–D68)

After the OpenAI account deactivation froze the pinned GPT-5.5 env, all arms were
re-evaluated on an interim Opus-4.8-backed WM env (same judge; within-env comparisons
only; n=40 clean each): base 0.600/0.634, GRPO-smoke 0.600/0.643, SDPO-n8 0.550/0.596,
GRPO-v2@0090 0.550/0.590, GRPO-v2@0097 0.575/0.610.

**Cross-env paired Δ vs base** — GRPO-smoke: +0.101 (GPT-5.5, replicated n=79) and
+0.009 (Opus), the only arm non-negative on both; SDPO-n8: +0.102 but −0.045 on Opus
(env-specific gains); stabilized v2 scale checkpoints never beat the smoke on either env.

**📌 Family-best arm: GRPO, smoke checkpoint (`wm_tau_grpo_0020`), pinned-protocol row
0.658 (n=79).** Two findings ride along: (1) arm separations are eval-env-sensitive —
the motivating exhibit for the fidelity→transfer curve; (2) at tau scale the short
regime (20 scenarios × 1 epoch) is the sweet spot — more training, even fully
stabilized (KL ≤3e-3, zero collapse across 97 scenarios), does not improve transfer.

## Reward-integrity audits (AGENTS rule 12)

- **Training rollouts** (haiku WM + judge): within a group with identical tool
  *sequences*, the judge scored 0.95 vs 0.30 by *outcome* (right vs wrong booking, named
  in the critique). 1.0-reward rollouts executed the task's mutating action; low-reward
  rollouts did lookups then quit — critiques say exactly that.
- **Eval episodes** (GPT-5.5 WM + Opus judge): top-reward episodes are verified
  multi-step resolutions (diagnose → act → in-environment verification, e.g. speed test
  confirming a data refuel); zero-reward episodes are genuine policy mistakes
  (transfer-instead-of-decline; unauthorized cancellation) with the judge citing the
  specific policy grounds. No verbosity/sycophancy gaming pattern observed in either.

## Caveats (stated plainly)

- n=40/checkpoint (airline n=14): single-domain deltas carry wide CIs; the +10 overall
  is directional evidence, not a tight estimate. Pass-to-pass variance observed
  (a partial pass's clean subset read 0.609 for GRPO).
- Eval env is itself a world model (GPT-5.5-backed). Family-different from the training
  WM, but a real-env spot check remains the follow-on validation for a positive result.
- Judge = Opus 4.8 across arms; identical for every row, so relative deltas stand even
  if the judge has absolute biases.

## Ops appendix (what it took — full ledger in DECISIONS.md D34–D57)

Trainer: FSDP2 dangling lm_head pointer; CPU-offload bake; entropy_checkpointing
full-logits gate; expanded-zero-grad contiguity (upstream verl bug); target_modules
schema narrowed vs upstream. Serving: cross-region + cross-geo judge waterfalls (a
US-wide Opus brownout mid-eval); score timeout ≥ the full waterfall budget; never bounce
the eval WM under a live sequential pass. Checkpoints: drained LoRA adapter →
merge-into-text-model → splice-over-base-snapshot → vLLM with `qwen3_xml` parser.

## Kimi replication (gui-tasks, #117 corpus) — head-to-head complete

Env of record: **haiku-kimi-era** (D86 re-pin — haiku-backed gui WM, measured fidelity
0.714 vs sonnet-5's 0.581; temp-0 env steps, rubric judge, no seed_state by corpus
design). Same env/config/judge for every row; policy Qwen3.5-9B, n=80 episodes/row
(40 eval scenarios × 2), all clean. Raw records: box-7
`~/output/b3_kimi_haikuera_{base,grpo150,sft}.jsonl`; wandb `kimi-haikuera-*`.

| arm | success | mean reward | vs base |
|---|---|---|---|
| base | 0.113 | — | — |
| GRPO full-train (150 scen × n=8, 267 steps) | 0.250 | — | **+13.7 pts; paired +0.100** (13W-12L, n=40) |
| **offline SFT on the Kimi-K2.6 demonstrations (LoRA, 776 eps)** | **0.8375** | **0.867** | **+72.5 pts** |

**The kimi answer as measured in-WM: offline SFT dominates on-policy WM training by a
wide margin.** Honest caveats, cutting both ways:

1. **Evaluation collusion** — the eval WM is built from the *same* Kimi-K2.6
   demonstration traces the SFT policy imitates; a WM with memorized dynamics likely
   flatters imitation policies. Only a real-env row could resolve this, and kimi has
   none (no macOS harness; not building one per D78). SFT episodes were audited:
   legitimate multi-step completions, not WM-echo artifacts.
2. **Asymmetric information** — SFT distills a frontier CUA teacher; GRPO self-improves
   a 9B model against its own rollouts. Different information sources, not a pure
   algorithm comparison.
3. Training-env == eval-env for GRPO (haiku both) — caveat stated at the re-pin (D86);
   partially offset by the SFT comparator sharing the exact same eval env.

Synthesis for the writeup: on-policy WM training delivers a real but modest gain where
no teacher exists (+13.7); when frontier demonstrations exist, imitation is far
stronger at least in-WM. The natural next question — SFT-then-GRPO stacking — is out
of sprint scope (D78).

## Real-env row: GRPO smoke ckpt on real tau2 (D86.3) — negative, mechanism identified

Protocol: B2's `tau_real_eval.py` harness — the same 20 pinned eval scenarios resolved
to real tau2 (domain, task_id) via provenance, real domain tools over the real JSON DB,
real tau2 grader, pinned Opus 4.8 user-simulator, temperature 1.0, 2 trials/task (n=40).
Comparator: B2's base real row of record (0.900, n=40, same protocol). Raw records:
`.agents/docs/research/real_tau_eval_results/real_tau_grpo0020_b3.jsonl`.

| arm | real success | airline (n=14) | retail (n=16) | telecom (n=10) |
|---|---|---|---|---|
| base (B2, row of record) | **0.900** | 13/14 | 13/16 | 10/10 |
| GRPO smoke ckpt (in-WM 0.658) | **0.600** | 9/14 | 10/16 | 5/10 |

**Real Δ = −0.300 (paired per-scenario: −0.300, 0W-10L across 20 scenarios).** The
in-WM +10 does not survive contact with the real environment — it inverts.

Failure decomposition:
- **6/40 episodes (15 pts of the 30-pt gap) are deployment-format failures**: the
  checkpoint emits reasoning-only replies (all text inside the think channel, empty
  content, no tool call) on specific hard prompts — 5 of 10 telecom episodes plus one
  retail. tau2's strict message validation rejects them (retried 4×/task across two
  passes: 100% persistent). Scored as reward 0 per the RFT2 precedent (D82): these are
  policy failures at the deployment boundary, not infra. Base had ZERO such failures on
  the identical harness — the behavior is WM-training-acquired. This is the program's
  **third instance of the deployment-format channel** (RFT2 corrupted tool syntax,
  terminal empty-commands) and the mildest: the model still solves tasks (24 real
  successes), it just intermittently swallows its answer into the reasoning channel.
- The remaining 15 pts are genuine real-env underperformance spread across domains
  (scored-only success 0.706 — still 19 pts below base).

Program synthesis (tau, final): GRPO's +10 in-WM (gpt-5.5 era, replicated n=79) was
already known to compress to ≈0 under eval-env swaps (opus/azure eras); the real-env row
resolves the question — **the smoke ckpt's in-WM gain is not real transfer**. Combined
with B2's terminal result (WM-trained −0.042 real vs SFT −0.163) the honest cross-
benchmark picture is: WM training's real-env value is benchmark-dependent, and
in-WM deltas systematically overstate it unless the WM's action interface is as strict
as deployment (D86.5 queued substrate fix).

### ⚠️ CORRECTION (2026-07-15, same day): the row above violated the pinned harness config

The 0.600 row was served **with** `--reasoning-parser qwen3`; the pinned real-gym config
(D70, the config under which base 0.900 was measured) is **without** it — tau2's strict
message validation rejects pure-think turns (B2 saw 10/14 sims die the same way), and
think-in-content lifts both rows ~20pts (base 68.6%→90.0%). Consequences:

1. The "reasoning-only deployment failures" above are **not a novel finding** — they are
   B2's documented D70 gotcha, reproduced by mis-serving. Struck as a claim.
2. 0.600 vs 0.900 is not a valid comparison (config mismatch worth ~20pts on base).
3. The row is being re-run under the pinned config (no reasoning parser, think-in-content,
   same everything else) as `b3_grpo0020_v2`; the table below supersedes the one above.

Lesson recorded: before running any cross-chat-comparable row, grep DECISIONS.md for the
harness's pinned serving config — the protocol lives there, not in the harness's --help.

### Real-env row OF RECORD (v2, pinned D70 config — supersedes the tables above)

`b3_grpo0020_v2`: same harness/scenarios/user-sim, vLLM served per the pinned config
(no reasoning parser, think-in-content). n=40, zero errors. Raw:
`.agents/docs/research/real_tau_eval_results/real_tau_grpo0020_v2_b3.jsonl`.

| arm | real success | airline (n=14) | retail (n=16) | telecom (n=10) |
|---|---|---|---|---|
| base (B2, row of record) | 0.900 | 13/14 | 13/16 | 10/10 |
| **GRPO smoke ckpt (in-WM 0.658)** | **0.900** | 12/14 | 14/16 | 10/10 |

**Real Δ = 0.000 (paired per-scenario +0.000, 2W-2L of 20).** The tau in-WM +10 does
not transfer to the real environment — and does not hurt either. This closes tau's
scoreboard line: **trained ✓ / in-WM +10 (gpt-5.5 era, n=79, replicated) / real Δ 0.000.**
Consistent with the cross-env compression already documented (opus/azure eras ≈ base)
and with B2's R++ real row (+0.025): at smoke scale, WM-trained tau checkpoints are
real-env-neutral. The first-pass −0.300 row above stands only as provenance for the
config lesson (D70 pinned config; ~30pts of artifact from one serving flag).

## Cross-era compression is the central caveat on every in-WM delta (D90.7)

The tau headline arms, measured on every eval environment the program had:

| eval environment | base | GRPO smoke | SDPO-n8 | GRPO delta |
|---|---|---|---|---|
| GPT-5.5-era WM (results of record, n=79 pooled) | 0.550 | **0.658** | 0.650 | **+10.8** |
| Opus-era WM (interim, references re-evaled) | 0.600 | 0.600 | 0.555 | **0.0** |
| Azure GPT-5.5-era WM (third era, same base model, new serving stack) | 0.675 | ≤ base | ≤ base | **≤ 0** |
| **Real tau2 gym** (pinned D70 config, n=40) | 0.900 | 0.900 | — | **0.0** |

The +10.8 exists on exactly one eval environment and compresses to ≤0 on every other,
including reality. **Open hypothesis, stated plainly and not resolved here: the in-WM
gain may partly be judge/env-pleasing rather than task competence.** The mechanism is
available: training rewards and the GPT-5.5-era eval rows share reward machinery from
the same model-family lineage, so GRPO can ascend idiosyncrasies of that grader — and
every grader that differs from the training-reward stack (Opus env, Azure serving
stack, tau2's real rule-based grader) shows no lift. Points against it: the training
WM was haiku (different family from the GPT-5.5 eval env, the strongest circularity
blunting we had), reward-integrity audits found no gaming pattern in episode text, and
the real-env row shows no *harm* either (a pure grader-hack usually costs real
performance — cf. terminal offline-SFT at −0.163). n=40–79 also leaves room for plain
noise. **This is the primary open question the program's in-WM deltas carry**; the
pre-registered discriminator is B2's D90.1 confirmatory pass on the untouched tau
val-split (one pre-declared pass, no reruns), and the queued substrate fix (D86.5:
WM action-parser strictness = deployment strictness) closes the adjacent loophole.

### ⚠️ D90.6 CORRECTION (2026-07-16): the kimi rows deviated from the D73 pin, undeclared

What was actually run in every kimi row (haiku-era AND the superseded sonnet-era):
the **first 40 scenarios — a positional prefix in pin-file order — of the pinned
100-scenario eval set, × 2 episodes each** (80 eps/row). Verified against the pin file:
the scenarios are exactly indices 0–39. Source of the deviation: the tau protocol shape
(`--num-scenarios 40 --epochs 2`) carried over into the kimi eval chain instead of the
D73 gui protocol (ALL 100 scenarios × 1 episode). The subsample was positional, not
random, and was not declared. All three arms (base/GRPO/SFT) share the identical
deviation, so within-era paired comparisons remain internally consistent — but the rows
do not match the pin. Remediation per D90(6): full-100 rerun (`kimi-haikuera100-*`,
100 × 1, same env/config/judge) — those rows supersede the n=40 rows as rows of record
when they land; the n=40 rows above stay as correction-trail provenance.

### D90.6 RESOLUTION (2026-07-16): full-100 rows landed — the kimi GRPO headline FLIPS

Rows of record (pinned D73 protocol: ALL 100 eval scenarios × 1 episode, haiku-kimi-era,
same env/config/judge as the n=40 rows; SFT with enable_thinking=false). Raw records
committed: `.agents/docs/research/kimi_eval_results/b3_kimi_haikuera100_*.jsonl`;
wandb `kimi-haikuera100-*`.

| arm | success | mean reward | paired vs base |
|---|---|---|---|
| base | 0.330 | 0.364 | — |
| GRPO full-train (150×8) | 0.240 | 0.286 | **−0.079 (33W-43L of 100)** |
| **offline SFT (Kimi-K2.6 demos)** | **0.830** | **0.882** | **+0.518 (74W-7L of 100)** |

1. **The "+13.7 / first positive kimi Δ" headline was a subsample artifact and is
   WITHDRAWN.** On the full pinned set GRPO is *negative* (paired −0.079). The first-40
   prefix happened to be far harder for base (0.113 there vs 0.330 full-set), flattering
   the trained arm. Combined with the strong in-run training curve (rolling 37%→85% on
   train scenarios), the honest mechanism is **overfitting to the training distribution
   with negative transfer to held-out scenarios** — consistent with the program's
   in-WM-fragility finding (cross-era table above), now demonstrated across scenario
   subsets as well as grader swaps.
2. **SFT's dominance is robust to the protocol fix** (0.8375 on the 40-subset → 0.830
   full-set; paired +0.518, 74W-7L) — the evaluation-collusion caveat still applies to
   its absolute level, but its ranking above base and GRPO is not subsample-sensitive.
3. Kimi scoreboard line of record: **trained ✓ / in-WM Δ (pinned protocol) GRPO −0.079,
   offline SFT +0.518 / real-env Δ n-a (no macOS harness).**
