# BENCH-B — ICL arms (coordinator): results

The no-gradient arms of the six-method ladder: can Qwen3.5-9B improve on held-out tau scenarios
purely from judge-critique context, with the world model as the environment? Protocol = the
pinned shared eval (DECISIONS.md D30/D33): `scenarios_eval.jsonl` (all 20) × 2 episodes,
policy temp 1.0, max_steps 20, env = pinned GPT-5.5 WM, reward judge = Opus 4.8 (us-east-1).
Policy served from standalone vLLM 0.17 on h100-dev-box-8 (`--reasoning-parser qwen3`, no tool
parser — the runner parses JSON/Qwen-XML off raw text). Base row is B2's (`wm_tau-eval-base-v3`)
per D30 — not re-run.

## Rows

| arm | success | mean reward | n | notes |
|---|---:|---:|---:|---|
| base Qwen3.5-9B (B2) | 55.0% | 0.568 | 40 | shared base row |
| ICL-single attempt-1 (ours) | 52.5% | 0.539 | 40 | independent base replication — consistent with B2 |
| **ICL-single (attempt-2, post-critique)** | **45.0%** | **0.512** | 40 | test-time critique retry: NET NEGATIVE |
| **ICL-multi (cross-task memory)** | **62.5%** | **0.643** | 40 | the only arm above base so far |

Costs: single $7.40 target-side / 40 rows (80 episodes incl. retries); multi $8.56 / 40 rows;
collect $25.04 / 97 scenarios. Judge cost tracked separately per D12.

## Finding 0 — cross-task critique memory works, and the win is within-domain (ICL-multi)

62.5% / 0.643 vs base 55.0% / 0.568 (n=40 each; within the ±15pt single-row CI, so the paired
per-scenario analysis vs B2's base_v3 records is the instrument): **+0.075 mean paired reward,
7 wins / 5 losses / 8 flat — and the entire gain concentrates in airline (+0.250 paired),
the domain where the memory holds 36 same-domain records.** Retail (-0.028) and telecom
(-0.005) are flat. Also notable: parse_failures dropped to 0 (from 11 in single / 5 in the
haiku shakeout) — the memory examples appear to anchor the output format as a side effect.

Read together with Finding 1: Qwen3.5-9B cannot use a critique to repair its OWN failed
attempt (single, negative), but aggregated critiques of OTHER tasks in the same domain do
transfer (multi, positive). Learning-from-feedback works across tasks, not within retries —
at v1 scale, cheap cross-task context beats every gradient method tried so far
(SFT 60.0%, PPO 54.1%, R++ 52.5%), which sharpens exactly what GRPO/SDPO must beat.

## Finding 1 — test-time self-critique makes Qwen worse (ICL-single)

Paired across the 40 rows: attempt-2 (with the judge's critique of attempt-1 in context)
improved 11, worsened 10, flat 19 — but regressions were larger than gains
(success 52.5% → 45.0%, reward 0.539 → 0.512). Retail collapsed (18.75% at attempt-2).

Failure-mode audit (worst regressions, critiques read by hand): after a NEAR-MISS attempt-1
(reward 0.75–0.85 — right trajectory, one flaw), the critique-primed retry **stops early**:
it re-executes the diagnosed step then quits before performing the required actions
(scenarios 37, 38: info gathered, exchange/return never called), or skips the verification
the critique itself demanded (39: cancels without identity check). The critique induces
second-guessing and truncation, not targeted repair — the same "feedback redistributes
behavior rather than adding competence" mechanism B2 found in SFT weights, here in context.
Directly relevant to SDPO expectations: its mechanism is critique-to-learning-signal via
gradients; this row shows the same model cannot do that conversion in context.

## Finding 2 — format friction is real (D31 corroboration)

parse_failures: 11 across the single row's ~80 episodes and 12 across collect's 97 (each
recovered or terminated per the one-nudge policy and counted per-row) vs 5/40 for the haiku
stand-in. Qwen3.5-9B fights the output format; without the Qwen-XML fallback parser these
rows would measure the parser.

## Collect pass (memory construction; not an eval row)

97 pinned train scenarios, one episode each vs the GPT-5.5 WM: 87 scored (51.7% success —
train-scenario difficulty is comparable to eval), 10 infra-error rows (Bedrock Opus judge
throttled during the tail when collect and single ran concurrently; episodes completed,
scoring failed, rows marked `error` and EXCLUDED — no fake zeros, no memory records).
Memory = 87 records (36 airline / 51 retail; zero telecom by construction, see D26 —
telecom train tasks were all leakage-filtered).

## Caveats (apply to every arm's rows)

- Telecom eval scenarios saturate for strong policies (D35): ~725 near-duplicate captures of
  those 5 tasks sit in the WM's retrieval buffer — report per-domain, don't read telecom as
  generalization.
- n=40/arm ⇒ CI ≈ ±15pts (B2's estimate): single-row differences within ~10pts are not
  individually significant; the paired per-scenario analysis is the sharper instrument.
- The eval env is a world model, not Sierra's tau2 gym. Same env for every arm, so the
  LADDER is apples-to-apples; absolute numbers are wmh-relative. (Real-env spot-check is the
  designated follow-up if any arm separates.)

## Reproduction

```bash
# policy server (box-8): PATH must include /data/venv/bin (ninja — JIT compile crashes without it)
# corpus: fetch first via environment_capture.hub.fetch_corpus('tau-bench')
vllm serve Qwen/Qwen3.5-9B --port 8001 --reasoning-parser qwen3 ...   # see D47
ssh -f -N -L 8011:localhost:8001 h100-dev-box-8

uv run python packages/environment-capture/tau-bench/rl/icl.py --mode collect --scenarios train --wm gpt-5.5 \
    --policy "vllm:Qwen/Qwen3.5-9B@http://localhost:8011/v1"
uv run python packages/environment-capture/tau-bench/rl/icl.py --mode single --scenarios eval \
    --episodes-per-scenario 2 --attempts 2 --wm gpt-5.5 --policy "vllm:Qwen/..."
uv run python packages/environment-capture/tau-bench/rl/icl.py --mode multi --scenarios eval \
    --episodes-per-scenario 2 --wm gpt-5.5 --policy "vllm:Qwen/..."
```

Raw per-row records (actions, rewards, critiques, costs): `.agents/docs/research/icl_eval_results/icl_*_gpt55_qwen.results.jsonl`. Ops gotchas hit standing this up: vLLM 0.17 dropped
`--disable-log-requests`; ssh-launched vLLM dies without `setsid` + stdin redirect AND
crashes on first inference if PATH lacks `ninja`; local port 8001 was already tunneled to
BENCH-A's AgentWorld server (use another local port).
