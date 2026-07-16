# RL smoke harness (tau-bench)

`smoke.py` exercises every wmh-side data path each downstream training chat will consume
end-to-end against the real tau-bench world model on Bedrock, at tiny scale (~30 haiku calls,
`max_steps=3`, 2 scenarios). It exists so we find interface problems in the RL seam here
before the transfer prompts spawn four training chats that would each hit the same wall.

## What it checks

Six paths, each independently PASS/FAIL:

1. **ICL** — one closed-loop episode via `run_episode(WorldModelEnv(wm), agent, task)`, then a
   scored follow-up rollout so we can build the augmented next-episode prompt that folds the
   reward-judge critique back in. This is the whole ICL story: no gradient, just critique-in-prompt.
2. **SFT** — chat-format dataset builder over two recorded train traces, one JSON example per
   step: `{messages: [system, user], completion: "<recorded tool call JSON>"}`. Shape only; no
   training.
3. **PPO / REINFORCE++** — n=1 rollout per scenario stepped directly against the WM
   (`new_session -> agent loop -> score_session -> end_session`), producing a batch of
   `{task_id, rollout_steps, reward, success, step_rewards}`.
4. **GRPO** — n=2 rollouts per scenario, grouped by `task_id`; group-relative advantages
   (`reward - group_mean`) computed and printed.
5. **SDPO** — one scored rollout emitted as a `{task, rollout, critique, reward}` feedback record
   where `critique` is `EpisodeScore.critique` (SDPO's tokenized teacher signal).
6. **HTTP** — the FastAPI seam claas-verl will actually call: `create_app` + `TestClient` in
   process, driving `session -> step -> score -> delete`.

## Run

```bash
uv run python packages/environment-capture/tau-bench/rl/smoke.py   # from repo root; AWS default profile, us-east-1
```

The script loads `packages/environment-capture/tau-bench/models/tau-bench/` (built config + retrieval index + optimized
prompt) but overrides the runtime provider to Bedrock haiku
(`us.anthropic.claude-haiku-4-5-20251001-v1:0`) for BOTH the serve model and the reward judge, so a
smoke doesn't cost Opus money. All 6 paths must PASS.

## Verified output

```
wmh RL smoke: loading tau-bench world model with Bedrock haiku (us-east-1)...
  loaded wm + 822 train traces + 2 scenarios (0.2s)

--- 1_ICL ---
scenario.task[:120]: '{"domain": "airline", "known_info": "You are Anya Garcia.\n\nYour user id is: anya_garcia_5901.\n\nYour confirmation num'
  stop_reason=max_steps steps=3
  scored followup rollout: reward=0.10 success=False
  critique[:200]: 'You retrieved user details but failed to complete the task. ...'
[PASS] 1_ICL: episode steps=3, follow-up reward=0.10, critique injected

--- 2_SFT ---
  dataset size: 9 examples
[PASS] 2_SFT: 9 SFT examples built from 2 train traces

--- 3_PPO_REINFORCE ---
  rollout: {'task_id': 'f9203b81...', 'rollout_steps': 3, 'reward': 0.6, 'step_rewards': [0.8, 0.9, 1.0]}
  rollout: {'task_id': '0af7f822...', 'rollout_steps': 3, 'reward': 0.3, 'step_rewards': [0.4, 0.3, 0.3]}
[PASS] 3_PPO_REINFORCE: 2 single-rollout PPO/R++ examples

--- 4_GRPO ---
  group f9203b81...: mean_reward=0.300
    rollout 0: reward=0.30 adv=+0.000
    rollout 1: reward=0.30 adv=+0.000
  group 0af7f822...: mean_reward=0.150
    rollout 0: reward=0.15 adv=+0.000
    rollout 1: reward=0.15 adv=+0.000
[PASS] 4_GRPO: 2 groups, 4 rollouts, group-relative advantages computed

--- 5_SDPO ---
  SDPO record (critique[:200]): 'You gathered relevant information but failed to complete the task as instructed. ...'
  rollout_steps=3 reward=0.20
[PASS] 5_SDPO: 1 SDPO feedback record built (rollout_steps=3)

--- 6_HTTP ---
  session_id=cc154e34b4a24d11bec122fbc44aef49
  step observation.content[:200]: '```json\n{\n  "output": {\n    "reservation_id": "EHGLP3", ...'
  score: reward=1.00 success=True
  final usage: {'total': {'calls': 2, 'input_tokens': 4118, 'output_tokens': 574, 'cost_usd': 0.006988}, ...}
[PASS] 6_HTTP: HTTP session cc154e34 ok (reward=1.00)

=============================
SMOKE SUMMARY
=============================
  1_ICL             PASS   episode steps=3, follow-up reward=0.10, critique injected
  2_SFT             PASS   9 SFT examples built from 2 train traces
  3_PPO_REINFORCE   PASS   2 single-rollout PPO/R++ examples
  4_GRPO            PASS   2 groups, 4 rollouts, group-relative advantages computed
  5_SDPO            PASS   1 SDPO feedback record built (rollout_steps=3)
  6_HTTP            PASS   HTTP session cc154e34 ok (reward=1.00)
=============================
```

## Notes for the transfer-prompt chats

- **`run_episode` vs `WorldModel.score_session`.** `run_episode` opens its OWN session inside
  the `WorldModelEnv` and closes it on return, so you cannot call `wm.score_session(session.id)`
  on that same session afterwards. RL rollouts that need `score_session` should step the WM
  directly instead: `new_session -> agent.act loop -> score_session -> end_session`. That
  pattern is `_rollout_direct` in `smoke.py`. `run_episode` is still the right primitive for
  offline eval (compare against the real env) — just not for reward collection.
- **`_reward_provider` is currently module-private on `WorldModel`.** `WorldModel.load` only
  accepts one `provider`; if you want the reward judge on a different (cheaper) model you have to
  poke `wm._reward_provider` after loading. Consider promoting this to a `WorldModel.load(...,
  reward_provider=...)` kwarg before the training chats start; today's smoke works around it.
- **Bedrock model ids need the dated suffix on inference profiles.** `us.anthropic.claude-haiku-
  4-5` alone returns `ValidationException`; the full profile id
  `us.anthropic.claude-haiku-4-5-20251001-v1:0` works. `wmh.tracking.pricing` already strips the
  suffix for the price lookup, so cost accounting still works.
- **Identical GRPO rollouts under temperature=0.** The default `provider.complete` is deterministic,
  so both rollouts in a group land on identical outputs and advantages collapse to 0. That is fine
  for wiring smoke; real GRPO training must pass a non-zero temperature (or top_p) so the group
  actually varies.
- **`scenarios_from_traces` first-seen order.** For a tau-bench train split it picks whatever
  trace hashed lowest first. If a downstream chat wants a specific scenario, filter/sort by
  `provenance` rather than assuming index 0 is stable across runs.

## Results

The BENCH-B2 results-of-record narrative (all arm tables, findings, failure analysis)
lives at `.agents/docs/research/bench_b2_results.md`; the finished fidelity→transfer
writeup is promoted to `docs/research/fidelity_transfer_curve.md`.
