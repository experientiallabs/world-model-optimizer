# Distill mode (`wmh optimize harness <agent> harbor --mode distill`)

The other optimizer modes edit the agent's *harness*; distill mode trains the agent's *model*.
It runs on-policy distillation of a Tinker LoRA student: harbor's own `terminus-2` agent rolls
out on real harbor benchmark tasks while sampling from the student's current weights; a larger
teacher model scores the exact tokens the student sampled; and each training
step nudges the student toward the teacher with a per-token reverse-KL objective (the
teacher-minus-student logprob gap as the advantage, trained under Tinker's `importance_sampling`
or `ppo` loss). A holdout gate at the end compares teacher, student-before, and
student-after solve rates, and only an adapter that closes enough of the gap to the teacher is
promoted. The result is a small model that behaves like the big one inside your agent, plus a
ready-to-paste serving snippet.

The same run can also train **off-policy**: with `[offpolicy] epochs > 0`, the teacher rolls
out on the train split, its kept trajectories merge into datums through the identical prefix
merge, and the student trains hard-target cross entropy over them for `epochs` passes at
`minibatch_datums` datums per optimizer step, before any on-policy step runs. Off-policy is the
cheaper objective (the corpus is collected once, and every epoch after that costs no rollout
spend) and the exposure-biased one (the student learns states the teacher visited, not its
own), so the two modes are read against each other. The phase is resumable at datum
granularity, so an interrupted run continues at the next minibatch instead of paying for the
whole schedule again.

## Prerequisites

- **The distill extra**: `uv sync --extra distill` (installs the `tinker` SDK and `wandb`).
- **`TINKER_API_KEY`** in the environment: the student trains and samples on Tinker, and the
  teacher scores there too.
- **`E2B_API_KEY`** when the run config sets `harbor.backend = "e2b"` (rollout trials in E2B
  sandboxes); `backend = "local"` runs them on your machine instead.
- **Free E2B sandbox capacity** for `backend = "e2b"`: a running trial holds one concurrent
  sandbox (harbor's task environment; terminus-2 itself runs in the `wmh` process), so a
  run needs `train.trial_concurrency` free slots against your account's concurrent-sandbox
  limit (100 by default; set `WMH_E2B_SANDBOX_CAP` when yours differs). See
  [Sandbox capacity](#sandbox-capacity).
- **A harbor job template**: the Harbor `JobConfig` YAML/JSON naming the benchmark dataset the
  trials run against, pointed at by the config's `[harbor] job_template`.
- **Task-id splits**: two JSON files, each a plain array of task-id strings. The train split
  feeds rollouts and interim evals; the holdout split (disjoint, enforced) is reserved for the
  baselines and the promotion gate.

## The run config

One TOML file describes one run. `[student]`, `[teacher]`, and `[harbor]` are required; every
other section has complete defaults. A minimal, realistic config:

```toml
[student]
base_model = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"  # the Tinker LoRA student
lora_rank = 32

[teacher]
model = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"    # scores student tokens

[harbor]
job_template = "tb2-job-template.yaml"  # Harbor JobConfig for the benchmark
backend = "local"                       # or "e2b" (needs E2B_API_KEY)

[rollout]
max_turns = 100                # per-episode turn cap (terminus-2's max_episodes)
episode_timeout_s = 1800.0     # per-episode wall budget, applied as harbor's
                               # agent-phase timeout (terminus-2 has none of its own)
context_budget_tokens = 65536  # episodes that outgrow this are dropped whole;
                               # also terminus-2's TinkerLLM context_limit, so keep
                               # it sampling.max_tokens below the served window or a
                               # full-budget call exceeds it

# REQUIRED for any REASONING student or teacher. Terminus-2 keeps only the parsed text of
# each assistant turn, and the auto-discovered renderer of a reasoning model (nemotron3,
# nemotron3_ultra, qwen3_5, which also serves Qwen3.6) hands the terminus parser a list
# instead of a string, raising TypeError before the trial grades anything. Those renderers
# also strip the thinking block when they re-render a turn as history, so turn N+1's prompt
# no longer extends turn N's and every turn becomes its own datum fragment, at a cost
# quadratic in turn count (measured: 2.7x the tokens at 6 turns, 7.8x at 20, 15.1x at 40).
# The wmh verbatim renderers fix both: the parser sees the action text as a plain string,
# and history replays each turn from its exact sampled ids, so the reasoning is trained on
# and the episode stays one datum. Keyed per base model because the teacher's own rollouts
# (warmup, teacher baseline) sample a different one. A typo here fails at config load.
[rollout.renderers]
"nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16" = "wmh/nemotron3_verbatim"
"nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16" = "wmh/nemotron3_ultra_verbatim"
# Qwen3.5 and Qwen3.6 both use "wmh/qwen3_5_verbatim"; plain Qwen3 uses "wmh/qwen3_verbatim".

[train]
steps = 40           # optimizer steps
tasks_per_batch = 8  # tasks sampled per step
group_size = 4       # attempts per task (the on-policy group)
# loss = "ppo"       # optional: train the same per-token advantages under a
#                    # clipped-ratio surrogate, so the update is bounded by the
#                    # loss's ratio clip instead of by bounding the advantage
# loss = "topk_ce"   # optional: weighted CE over the teacher's top-k candidate
# topk = 8           # tokens per position instead of reverse-KL on realized
#                    # tokens; trains ~topk x the token volume per step
# advantage_clip = 4.0     # optional symmetric bound on the per-token advantage;
#                          # unset (the default) trains the raw gap
# center_advantages = true # optional batch-mean baseline; false (the default)
#                          # keeps advantage_mean readable as the mean gap

[sampling]
temperature = 1.0    # keep 1.0: issued logprobs stay comparable to the teacher
max_tokens = 8192

[offpolicy]              # optional off-policy distillation phase; epochs = 0 disables it
epochs = 4               # passes over the kept teacher trajectories
minibatch_datums = 8     # datums per optimizer step; 0 trains one full-batch pass an epoch
rollouts_per_task = 2    # teacher attempts per train task when collecting the corpus
keep = "passed"          # or "all" to train on the teacher's failures too
learning_rate = 5e-5     # unset uses train.learning_rate
shuffle_seed = 17        # optional per-epoch datum shuffle (reproducible; unset keeps order)
checkpoint_every = 1     # save state + rewrite the resume cursor every N optimizer steps
# trajectories_from = "runs/prior"  # reuse another run's collected corpus, charged nothing

[warmup]
steps = 0              # SUPERSEDED by [offpolicy] (same objective, full-batch only, no
rollouts_per_task = 2  # cursor). Setting both sections nonzero is rejected at load.

[eval]
every = 0  # interim evals off (the default); N > 0 evals a train subsample every N steps
# Baseline reuse: point these at a prior run's eval reports to skip re-running
# the holdout baselines. Validated before use: identical holdout task ids,
# attempts >= this run's gate.k, and the same teacher (teacher baseline) or
# student base model (student baseline, via the report's recorded base_model).
# teacher_baseline_from = "runs/prior/evals/baseline-teacher.json"
# student_baseline_from = "runs/prior/evals/baseline-student-before.json"

[gate]
k = 3                        # holdout attempts per task for each baseline
min_teacher_fraction = 0.7   # student-after must reach 70% of teacher solve rate
require_no_regression = true # and must not fall below student-before

[pricing]  # USD per million tokens, from the provider's price list
teacher_prefill = 2.49
teacher_sample = 6.225
student_prefill = 0.30
student_sample = 0.70
student_train = 0.60
# cached prefill rates default to 20% of the full prefill price

[budget]
max_usd = 600.0  # hard cap; the run checkpoints and aborts resumably at it

[tripwire]  # degeneration guards; every bound is a FRACTION of this run's own
            # baseline, measured at its first training step (never an absolute)
enabled = true
entropy_warn_frac = 0.5    # warn below half the baseline entropy/token
entropy_kill_frac = 0.3    # abort below 30% of it (after N steps in a row)
length_warn_frac = 0.5     # same pair for mean sampled tokens per episode
length_kill_frac = 0.25
kill_consecutive_steps = 2 # one short batch is a small task draw; two is a trend

[wandb]
enabled = true   # optional live tracking (needs WANDB_API_KEY or wandb login)
project = "wmh-distill"
```

Billing follows the provider's per-request model: every agent turn re-bills its whole prompt,
with the verbatim repeated prefix at the cached rate, so the projected volumes are much larger
than the unique context size. The CLI prints the per-meter projection and asks for
confirmation before anything is spent; unpriced meters print as `unknown`, and a run with
unpriced meters and no `budget.max_usd` refuses to start non-interactively.

## Running it

```bash
wmh optimize harness pi harbor --mode distill \
  --distill-config run.toml \
  --task-ids train-task-ids.json \
  --holdout-task-ids holdout-task-ids.json \
  --run-dir runs/distill-01
```

The agent argument works like the other optimizer modes: the literal `pi` is the built-in
default agent, and `name@ref` seeds from a stored harness version (the harness must be a
pi-node harness; it is pinned for the whole run). `--backend local|e2b` overrides the config's
`harbor.backend`. `--yes` skips the cost confirmation when the spend is accountable. Add
`--promote` to be offered a `[models.agent]` settings write after an accepted gate.

Before spending anything the run preflights: renderer resolution, a student/teacher tokenizer
fingerprint check, one-token pings, and a tokens-in-tokens-out (TITO) recompute proof that the
sampling and scoring paths agree on the student's own tokens. On `backend = "e2b"` it also
checks sandbox capacity first (below).

## Sandbox capacity

E2B caps concurrent sandboxes per account, and a harbor trial's task-environment sandbox lives
for its own multi-hour timeout. A run that dies without graceful shutdown (crash, SIGKILL,
budget abort, machine sleep) therefore leaves its sandboxes running, and the next run starves at
the cap: every trial fails at sandbox creation with
`RateLimitException: 429 ... maximum number of concurrent E2B sandboxes`, which surfaces as
trials producing zero token spans and looks exactly like a broken model.

Two mechanisms keep that from happening silently.

- Every sandbox a wmh run creates is recorded, with its owning process id, in a per-process
  JSONL ledger under the WMH user state directory (`$WMH_HOME/e2b-sandboxes`, else
  `~/.wmh/e2b-sandboxes`), and marked released when its kill is proved. A ledger entry whose
  owning process is gone is a provable orphan that can be killed by exact id.
- An e2b-backed distill run preflights capacity: it counts running sandboxes, auto-reclaims
  those provable orphans, and refuses to start when `2 x train.trial_concurrency` slots are
  still not free, naming the numbers instead of starving.

To inspect or reclaim capacity by hand:

```bash
wmh e2b reap                      # dry run: what is running, and what would be killed
wmh e2b reap --yes                # kill orphans of dead local runs (exact recorded ids)
wmh e2b reap --stale-minutes 60   # ALSO match harbor trial sandboxes account-wide by age
```

`--stale-minutes` matches on the account, not just this machine, so it can kill a run on another
machine or in another checkout; sandboxes whose local owner process is still alive are never
selected. Both forms are dry runs until `--yes`.

## What a run produces

Everything durable lands under `--run-dir`:

```text
<run-dir>/
  config.toml         # exact snapshot of the run config
  distill-run.json    # pinned CLI inputs (splits, backend, seed harness hash)
  offpolicy.json      # the off-policy phase's terminal record (its existence means done)
  offpolicy-cursor.json  # only while that phase is INTERRUPTED: the last checkpointed
                      # optimizer step and the tinker:// state saved with it
  metrics.jsonl       # one row per off-policy/warmup/training step: solve rate (over the
                      # trials that actually executed) and graded_solve_rate
                      # (the same trials at test resolution) plus
                      # scaffold_loss_rate,
                      # stop_reason_counts, infra_failed_trials, reverse
                      # KL/token, entropy_per_token and mean_generation_tokens
                      # with their baselines and ratios, the objective that
                      # trained the step (loss) and its advantage mean/std and
                      # clip fraction, datum and drop counts, truncated_spans,
                      # per-meter tokens, USD
  spend.json          # cumulative priced USD, updated on every charge
  checkpoints.json    # saved tinker:// training-state + sampler paths, plus the
                      # tripwire baseline (it has to survive --resume)
  evals/<name>.json   # baseline, interim, and student-after eval reports
  gate.json           # the promotion verdict
  model_card.json     # base model, teacher, artifact paths, gate record
  handoff.toml        # the [models.agent] serving snippet
  harbor/             # per-step harbor jobs dirs; each trial dir's result.json carries
                      # that trial's exact sampled token spans (rollout_details)
  eval-rollouts/      # isolated rollout root per eval batch
  warmup-rollouts/    # the teacher collection's rollout root, with its assembled trials in
  warmup-trials.json  # the corpus manifest both cross_entropy phases read and write
```

Read `scaffold_loss_rate` before reading any solve rate. It is the share of episodes that never
reached an explicit `submit`, with the per-reason breakdown in `stop_reason_counts`
(`max_turns`, `budget`, `no_tool_call`, `output_truncated`, `unparsed_tool_call`,
`provider_error`). Those episodes were cut off by the harness, so their rewards measure where the
cutoff fell rather than what the model can do: a high scaffold loss rate means the reported solve
rate is a property of the budgets, not of the student. Trials that never produced verifier
evidence at all (a sandbox that was never created) are counted separately as
`infra_failed_trials` and excluded from `solve_rate`, and an eval where NO trial executed refuses
to record a 0.0 rather than putting a null measurement behind the gate.

`graded_solve_rate` sits beside every solve rate, in `metrics.jsonl`, in each `evals/<name>.json`,
and on the dashboard (`train/graded_solve_rate`, `eval/<name>-graded`). It is the same trials read
at test resolution: harbor's pytest verifiers write a CTRF report per trial
(`<trial_dir>/verifier/ctrf.json`), and a trial's graded score is the share of its tests that
returned a passing verdict, so a run that fixes half of a task registers instead of scoring the
same 0 as one that did nothing. On a 48-episode TerminalBench-2 probe, 9 of 46 gradeable trials
(19.6%) scored `reward = 0` while passing at least one test, and the batch read 0.319 graded
against 0.217 binary. Three properties to keep in mind:

- **Binary stays the headline, and it is what the promotion gate reads.** Binary success is the
  benchmark's own definition; graded exists because a small holdout has no resolution to detect an
  effect with.
- **Graded is coarse, not continuous.** The probe's tasks carried 1 to 6 tests, so most trial
  scores land on 0, 1/2, or 1, and a single-test task is exactly as binary as its reward.
- **Its denominator is its own: `graded_trials`.** A trial with no readable CTRF report (a verifier
  that timed out wrote neither a reward nor a report) is excluded, exactly as it is excluded from
  `solve_rate`, never averaged in as a 0.0. A `graded_solve_rate` of 0.0 with `graded_trials = 0`
  is a null measurement, and tests that the grader skipped stay out of the per-trial denominator
  for the same reason (they do not block the binary pass either).

Read `entropy_ratio` and `generation_tokens_ratio` next to them. Reverse-KL distillation can
degenerate in ways the KL curve cannot show, because KL falls while the policy collapses: the
student folds into one mode, or, since pure KL gives EOS no gradient, never learns to stop.
Each step pools the student's own sampled logprobs into `entropy_per_token`
(`-mean(sampled_logprobs)` over the batch's loss positions, an unbiased single-sample estimator
of the policy's entropy, and a lower bound on the T=1 entropy whenever `sampling.temperature`
is below 1.0) and its sampled tokens into `mean_generation_tokens` (tokens per span-bearing
episode). Both are pooled over the whole batch, never per episode, because healthy episodes
vary enormously: on a 48-episode TerminalBench-2 probe the per-episode entropy ranged down to
0.082 nats/token and lengths ran from 349 to 30,869 tokens.

Both bounds are one-sided, on the downside; the runaway direction (never learning to stop)
shows as `mean_generation_tokens` rising, with `stop_reason_counts` and `scaffold_loss_rate` as
its sharper signals. The `[tripwire]` bounds are fractions of the baseline the run measures at
its own first training step, which is persisted in `checkpoints.json` so `--resume` reuses it instead of
re-anchoring on an already degenerated policy. A breach logs a warning naming the metric, the
baseline, the value and the ratio; `kill_consecutive_steps` breaches in a row at kill level
checkpoint the run and abort it with the exact resume command. Absolute thresholds are
deliberately impossible here: the healthy, untrained baseline of this project's own student is
0.181 nats/token, under the 0.2 absolute floor a sibling lane pre-registered, so an absolute
rule would fire before the first gradient step and get muted.

An accepted gate additionally saves the adapter as an immutable version under the project's
`.wmh/adapters/<agent>/vN/` with a movable `champion` alias, and prints the serving handoff:
a `[models.agent]` TOML snippet pointing at the final `tinker://...` sampler path through
Tinker's OpenAI-compatible endpoint (authenticate by setting `WMH_ENDPOINT_API_KEY` to your
Tinker API key). With `[wandb] enabled = true`, steps, evals, spend, and the gate summary
stream to a Weights & Biases run that resumes with the run dir.

## Resume and budget behavior

Training state checkpoints on a cadence (`train.save_state_every`) plus at every abort. If the
run hits `budget.max_usd`, it saves what it can and exits with the exact resume command; raise
the cap in the config and rerun with:

```bash
wmh optimize harness pi harbor --mode distill --run-dir runs/distill-01 --resume
```

A resume needs only `--run-dir`: the CLI reloads the pinned splits, backend, and seed harness
from `distill-run.json`, the config from the `config.toml` snapshot (an explicit
`--distill-config` wins, which is how you raise the cap), and prior spend from `spend.json`,
so a resumed run can never spend the budget twice. Recorded baselines and a finished
cross_entropy phase (`offpolicy.json`, or the legacy `warmup.json`) are reused, the step count
continues from the latest checkpoint, and conflicting explicit
flags are rejected rather than silently changing the run. An off-policy phase that was
interrupted mid-schedule resumes at datum granularity: `offpolicy-cursor.json` names the state
saved at the last checkpointed optimizer step and how many steps are already applied, so the
resumed session restores those weights, reuses the teacher corpus in `warmup-trials.json`
rather than collecting it again, and trains only the minibatches that are left. The tripwire
baseline in
`checkpoints.json` is reused verbatim as well: a resumed session that re-measured it would
anchor on the policy it just restored, so a collapse would become the new normal and the
tripwire would never fire again.

## Troubleshooting

| Symptom | Meaning and fix |
|---|---|
| TITO preflight failure (`TITO recompute disagreement ...`) | The sampling and scoring paths disagree on the student's own tokens, so training data would be corrupt; check that the sampler path matches the student base model and that the pinned `tinker` SDK version is unchanged. |
| Empty-batch abort (`... every trial produced zero token spans`) | Consecutive steps sampled no completions at all. Either the student provider or its sessions are failing upstream (check the runner logs for worker completion warnings), or, on `backend = "e2b"`, trials are dying at sandbox creation because the account is at its concurrent-sandbox cap (`wmh e2b reap`). Fix the cause, then `--resume`. |
| Every trial 429s at sandbox creation (`maximum number of concurrent E2B sandboxes`) | The account is at its concurrent-sandbox cap, usually because a crashed run's sandboxes are still running out their multi-hour timeout. Run `wmh e2b reap` to see what is holding the slots, then `wmh e2b reap --yes` (orphans of dead local runs) or `wmh e2b reap --stale-minutes N --yes` (account-wide by age). See [Sandbox capacity](#sandbox-capacity). |
| Start refused (`not enough free E2B sandbox slots ...`) | The capacity preflight found fewer free slots than `2 x train.trial_concurrency` even after reclaiming provable orphans; free slots as above, lower `train.trial_concurrency`, or raise the cap (`WMH_E2B_SANDBOX_CAP`). |
| Degeneration abort (`... consecutive training steps at a degeneration kill level`) | The student's own sampled tokens collapsed against the baseline this run measured at its first training step: entropy fell (mode collapse) or episodes got much shorter (EOS learned too eagerly), which reverse KL alone does not show. Read the breached ratio in the message, lower `train.learning_rate` or restart from an earlier checkpoint, then `--resume` (the recorded baseline is kept, so the resumed run is not re-anchored on the collapse). |
| A degeneration warning on a batch that was mostly infrastructure failures | The statistics are batch-pooled, so they tighten as usable episodes shrink; check `trials` against `empty_span_trials` and `infra_failed_trials` in the same row before believing it. Resampling the reference probe gave a healthy 32-episode batch a 1e-5 chance of a length warning and never once reached the kill bound, so a warning on a full batch is a real signal. |
| High `scaffold_loss_rate` in `metrics.jsonl` | Most episodes were cut off before the model declared itself done, so the solve rate reflects the budgets. Check `stop_reason_counts`: `max_turns` wants a higher `rollout.max_turns`, `budget` a higher `rollout.episode_timeout_s`, `output_truncated` a higher `sampling.max_tokens`, `provider_error` usually means prompts are overflowing the served context window (lower `rollout.context_budget_tokens` so it sits at least `sampling.max_tokens` below the window). |
| An eval refuses to record (`eval ... is a NULL measurement, not a 0.0`) | Every trial died before the verifier ran, so there is no solve rate to write. Almost always the concurrent-sandbox cap (a distill trial holds two sandboxes); see the E2B rows below, lower `train.trial_concurrency`, then `--resume`. |
| Deadline expiries (`TinkerDeadlineError: tinker <call> timed out after ...`) | A wedged Tinker session was cut off instead of hanging; transient ones retry with a fresh session on their own, and a persistent one can be given more headroom via the `WMH_TINKER_DEADLINE_<KIND>` env vars the error names. |
| Resume rejected (`LoadWeights can only be called on uninitialized models`) | Tinker accepts a checkpoint restore only on a model nothing has touched yet, so a resume loads its state as the very first call on a freshly created training client. If a restore is slow enough to blow its deadline, the run retries it on another fresh client; `WMH_TINKER_DEADLINE_LOAD_STATE` (600s by default) is how long one attempt may take, which large students may need raised. |
| Fragmentation warning (`N of M datum(s) are fragments ...`) | The agent edited its prompt history mid-episode, so shared context re-prefills at full price and teacher scoring multiplies; keep `rollout.compaction = false`, check `[rollout.renderers]` names a `wmh/*_verbatim` renderer for every reasoning model, and check `truncated_spans` in the same row. |
| Nonzero `truncated_spans` in `metrics.jsonl` | Turns sampled the full `sampling.max_tokens` and were cut off mid-answer. Nothing else reports this (harbor's own truncation guard cannot fire), so without this counter it reads as a model that writes broken actions. A truncated turn also cannot be replayed verbatim, so it fragments the episode. Raise `sampling.max_tokens`, keeping `rollout.context_budget_tokens` that far below the served window. |
