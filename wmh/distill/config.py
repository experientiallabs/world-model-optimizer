"""Per-run TOML configuration for an on-policy distillation run.

A distillation run is described by one TOML file with sections mirroring the
sub-models below (student, teacher, harbor, rollout, train, sampling,
offpolicy, warmup, eval, gate, pricing, budget, tripwire, wandb).
`load_distill_config` reads and
validates the file; `snapshot_toml` renders a validated config back to TOML so a
run dir can carry an exact snapshot of the configuration it ran with.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated, Literal

import tomli_w
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from wmh.distill.rendering import MISSING_DISTILL_EXTRA


class StudentConfig(BaseModel):
    """The Tinker LoRA student under training."""

    model_config = ConfigDict(extra="forbid")

    base_model: str
    lora_rank: int = 32


class TeacherConfig(BaseModel):
    """The teacher that scores student tokens, and how its vocabulary lines up.

    Two backends, and the backend choice fixes the token alignment:

    - `tinker` (the default) serves the teacher from the Tinker lineup and
      requires the teacher to share the student's tokenizer, so student token
      ids are scored verbatim (`alignment = "same_tokenizer"`).
    - `openai_compat` scores against a self-hosted vLLM OpenAI-compatible
      server, which is how a teacher outside the Tinker lineup (e.g. a
      quantized GLM checkpoint) is reached. Its vocabulary differs from the
      student's, so scoring goes through byte-aligned chunks
      (`alignment = "chunk"`) and the teacher's own `tokenizer` must be named.
    """

    model_config = ConfigDict(extra="forbid")

    backend: Literal["tinker", "openai_compat"] = "tinker"
    model: str
    checkpoint: str | None = None
    """Optional tinker:// checkpoint path to serve the teacher from
    (tinker backend only)."""

    endpoint: str | None = None
    """Base URL of the self-hosted vLLM OpenAI-compatible server serving the
    teacher (openai_compat backend only, where it is required)."""

    tokenizer: str | None = None
    """HF repo id of the teacher's tokenizer, e.g. `zai-org/GLM-5.2`. Required
    by the openai_compat backend: chunk alignment has to tokenize teacher-side
    text itself to map student token spans onto teacher token spans."""

    alignment: Literal["same_tokenizer", "chunk"] = "same_tokenizer"
    """How student token positions map onto teacher token positions.
    `same_tokenizer` scores the student's own ids directly; `chunk` splits each
    sampled span into byte-aligned chunks scored in the teacher's vocabulary."""

    @model_validator(mode="after")
    def _check_backend_axis(self) -> TeacherConfig:
        """Reject backend/alignment/field combinations that cannot be served.

        Returns:
            This config, unchanged, when the combination is coherent.

        Raises:
            ValueError: If the backend does not match the fields it needs (or
                forbids); every message names the key to add or drop.
        """
        if self.backend == "openai_compat":
            if self.checkpoint is not None:
                raise ValueError(
                    "teacher.checkpoint is a tinker:// path and only applies to "
                    'teacher.backend = "tinker"; drop teacher.checkpoint and point '
                    "teacher.endpoint at a server already serving the weights you want"
                )
            if self.endpoint is None:
                raise ValueError(
                    'teacher.backend = "openai_compat" needs teacher.endpoint: set it to '
                    "the base URL of the vLLM OpenAI-compatible server serving "
                    f'{self.model!r} (e.g. endpoint = "http://127.0.0.1:8000/v1"), or '
                    'switch to backend = "tinker" for a Tinker-lineup teacher'
                )
            if self.tokenizer is None:
                raise ValueError(
                    'teacher.backend = "openai_compat" needs teacher.tokenizer: set it to '
                    "the HF repo id of the teacher's tokenizer (e.g. tokenizer = "
                    '"zai-org/GLM-5.2") so chunk alignment can tokenize teacher-side text'
                )
            if self.alignment != "chunk":
                raise ValueError(
                    'teacher.backend = "openai_compat" requires teacher.alignment = '
                    f'"chunk", got {self.alignment!r}: a self-hosted teacher does not '
                    "share the student's vocabulary, so its logprobs cannot be read off "
                    "the student's token ids"
                )
            return self
        if self.endpoint is not None:
            raise ValueError(
                'teacher.endpoint is only for teacher.backend = "openai_compat"; the '
                '"tinker" backend reaches the teacher through the Tinker service, so '
                'drop teacher.endpoint or set backend = "openai_compat"'
            )
        if self.alignment != "same_tokenizer":
            raise ValueError(
                'teacher.backend = "tinker" requires teacher.alignment = '
                f'"same_tokenizer", got {self.alignment!r}: chunk alignment is only '
                'implemented for the "openai_compat" backend, so either drop '
                'teacher.alignment or set backend = "openai_compat" with an endpoint '
                "and tokenizer"
            )
        return self


class HarborConfig(BaseModel):
    """How rollouts are produced: harbor's terminus-2 agent on harbor tasks.

    Attempts per task are NOT configured here: training rollouts use
    `train.group_size` (the on-policy group) and evals use `eval.k` / `gate.k`.
    """

    model_config = ConfigDict(extra="forbid")

    job_template: str
    """Path to a Harbor JobConfig YAML/JSON used as the task template."""

    backend: Literal["local", "e2b"] = "local"
    reward_key: str = "reward"

    retries: int = Field(default=1, ge=0)
    """Harbor-level retries per failed trial. Distill batches see transient
    sandbox/runner deaths (e.g. an E2B transport drop killing the pi runner
    mid-episode); one retry absorbs them, and any trial that still ends
    without a verifier reward scores 0.0 instead of aborting the run."""


class RolloutConfig(BaseModel):
    """Per-episode rollout limits."""

    model_config = ConfigDict(extra="forbid")

    max_turns: int = Field(default=100, ge=1)
    """Episode turn cap; passed to terminus-2 as `max_turns` (f.k.a. `max_episodes`).

    100, not the harness-wide 20 that world-model closed-loop eval uses. Terminus-2 is unbounded by
    default (`_max_episodes = 1_000_000`) and TerminalBench-2 tasks routinely need 50 to 200 turns;
    at 20 the cap fired mid-tool-call on 45% of Ultra trials and 12% of Super trials, and every one
    of those scored reward 0. 100 is the point where a cap still bounds a looping agent (a real
    100-turn episode also runs into `episode_timeout_s` first) without being the thing that decides
    the score. Also pinned into the harness doc's `param:max-turns`, which no longer steers the
    rollout agent but still keys the per-candidate harbor job dir."""

    episode_timeout_s: float = Field(default=1800.0, gt=0)
    """Per-episode wall budget.

    Terminus-2 has no internal wall clock, so this is applied as harbor's own agent-phase timeout
    (`AgentConfig.override_timeout_sec`), which raises `AgentTimeoutError`; harbor swallows that
    and still verifies the work, so an episode cut off by the clock stays a graded trial. Without
    it every rollout and eval wave inherited the task's declared agent timeout. 1800s covers the
    real work in this suite (a single `apt-get install build-essential` observation was 45,131 chars
    and tens of seconds; the suite also compiles CompCert and boots QEMU) while staying inside the
    E2B sandbox lease ceiling (`MAX_EVAL_EPISODE_LIFETIME_S`, 3600s, minus one lease of cleanup
    headroom)."""

    context_budget_tokens: int = Field(default=65536, ge=1024)
    """Context cap: episodes where any call's prompt plus sampled tokens exceed
    it are dropped whole from training (`build_datums`), and the cost estimate
    caps per-episode tokens here.

    Also the context window the rollout agent's LLM is built with (terminus-2's
    `TinkerLLM(context_limit=...)`), so the agent measures its own prompts against the real
    serving limit instead of harbor's 32,000 default. Keep it at least `sampling.max_tokens`
    below the served window, or a full-budget prompt plus its output cap exceeds the window and
    the sampling call 400s."""

    renderers: dict[str, str] = Field(default_factory=dict)
    """Per-base-model override of the chat renderer terminus-2's `TinkerLLM` builds prompts with,
    keyed by the base model name (`student.base_model`, `teacher.model`). A model with no entry
    auto-discovers its renderer through
    `tinker_cookbook.model_info.get_recommended_renderer_name`.

    Keyed per model, not one value, because a run samples MORE than one base model: the student's
    rollouts, plus the teacher's own rollouts for the warmup collection and the teacher baseline
    eval. A Nemotron Nano student and a Nemotron Ultra teacher need different renderer names.

    Exists because the auto-discovered renderer of every REASONING model in this lineup is
    unusable under terminus-2 as the cookbook ships it, measured offline against the real
    tokenizers. Terminus-2 keeps only `parse_response(...)["content"]` in its chat history, and
    `nemotron3`, `nemotron3_ultra` and `qwen3_5` (the auto renderer for both Qwen3.5 and Qwen3.6)
    return that content as a LIST of thinking/text parts, which harbor's
    `TerminusJSONPlainParser` raises `TypeError` on, killing the trial before it grades anything.
    Those same renderers also strip the thinking block when they re-render an assistant turn that
    is no longer last, so turn N+1's prompt diverges from turn N's and EVERY turn becomes its own
    datum fragment, at a cost quadratic in turn count (2.7x the tokens at 6 turns, 7.8x at 20,
    15.1x at 40).

    What to name here: the wmh VERBATIM renderer for every Nemotron-3 or Qwen3.5/3.6 model in the
    run. `wmh.distill.renderers` wraps each reasoning renderer so the parser sees a plain `str`
    of the action text while the model's own reasoning is replayed into history token for token,
    which fixes both failures at once and keeps the episode a single datum:

        [rollout.renderers]
        "Qwen/Qwen3.5-9B" = "wmh/qwen3_5_verbatim"
        "Qwen/Qwen3.6-27B" = "wmh/qwen3_5_verbatim"
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16" = "wmh/nemotron3_verbatim"
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16" = "wmh/nemotron3_ultra_verbatim"

    The disable-thinking renderers (`nemotron3_disable_thinking`,
    `nemotron3_ultra_disable_thinking`) also hold the prefix property and parse, but they buy that
    by throwing the reasoning away, which is exactly the behavior distillation is meant to teach.
    Name one only when a run deliberately trains a non-reasoning policy."""

    @field_validator("renderers")
    @classmethod
    def _reject_blank_renderers(cls, value: dict[str, str]) -> dict[str, str]:
        """Reject blank or unknown renderer entries at config load, not at trial 1.

        Args:
            value: The raw `model -> renderer name` mapping.

        Returns:
            The mapping unchanged when every entry names a renderer the cookbook can build.

        Raises:
            ValueError: If either side of an entry is blank, or the renderer name is one
                `tinker_cookbook.renderers.get_renderer` does not know.
            ImportError: If the distill extra is missing, so no name can be checked.
        """
        for model, renderer in value.items():
            if not model.strip() or not renderer.strip():
                raise ValueError(
                    "rollout.renderers maps a base model name to a tinker-cookbook renderer "
                    f"name; got the entry {model!r} = {renderer!r}, which has an empty side"
                )
        if not value:
            return value
        # Deferred: wmh.distill.renderers subclasses cookbook classes at module scope, and
        # wmh.distill.config is imported by the CLI, which must work without the distill extra.
        try:
            from wmh.distill.renderers import VERBATIM_RENDERERS, is_known_renderer
        except ImportError as exc:
            raise ImportError(MISSING_DISTILL_EXTRA) from exc
        for model, renderer in value.items():
            if is_known_renderer(renderer):
                continue
            raise ValueError(
                f"rollout.renderers names the renderer {renderer!r} for {model!r}, which "
                "tinker-cookbook cannot build; a name that does not resolve would only fail on "
                "the run's first rollout. Use one of the wmh verbatim renderers "
                f"({', '.join(sorted(VERBATIM_RENDERERS))}) or a built-in cookbook name "
                "(qwen3, qwen3_5, nemotron3, nemotron3_ultra, their *_disable_thinking "
                "variants, ...)"
            )
        return value

    compaction: bool = False

    @field_validator("compaction")
    @classmethod
    def _reject_compaction(cls, value: bool) -> bool:
        """Reject compaction: it breaks the prefix property the cost model relies on.

        Every turn's prompt must extend the previous turn's tokens verbatim so
        prefill work amortizes across turns and sampled spans stay aligned for
        teacher scoring. Compacting mid-rollout rewrites the prefix, so each
        later turn would be a full re-prefill and issued spans would no longer
        appear verbatim in the episode tokens.
        """
        if value:
            raise ValueError(
                "rollout.compaction = true is not supported: compaction rewrites the "
                "token prefix mid-episode, breaking the prefix property that keeps "
                "prefill costs amortized across turns and sampled spans verbatim in "
                "the episode; set compaction = false (episodes that outgrow "
                "context_budget_tokens are dropped whole from training instead)"
            )
        return value


class TrainConfig(BaseModel):
    """Optimizer-loop schedule and batch shape."""

    model_config = ConfigDict(extra="forbid")

    steps: int = Field(default=40, ge=1)
    tasks_per_batch: int = Field(default=8, ge=1)
    group_size: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    loss: Literal["importance_sampling", "ppo", "topk_ce"] = "importance_sampling"
    """The distillation loss.

    `importance_sampling` (the default) trains per-token reverse-KL
    advantages over the student's realized tokens.

    `ppo` trains those same advantages (identical wire datums) under the
    service's clipped-ratio surrogate: the update is bounded by clipping the
    policy ratio rather than by bounding the advantage, which is the
    OpenClaw-RL / Slime formulation, so pair it with `advantage_clip` unset
    and `center_advantages = false`. The clip epsilon is the service default
    (reported symmetric 0.2); no `loss_fn_config` is sent. Note what the clip
    can and cannot do here: this loop takes ONE forward/backward plus one
    optimizer step per batch and refreshes the sampler every
    `sampler_refresh_every` steps, so at `sampler_refresh_every = 1` the
    ratio is ~1 for every token (only sampler-vs-trainer numerical drift,
    ~0.08 nat) and the clip almost never binds. It starts protecting the run
    when the policy drifts from the sampler that produced the batch (a larger
    `sampler_refresh_every`, or reused batches).

    `topk_ce` trains a weighted cross-entropy over the teacher's top-k
    candidate tokens at every loss position (renormalized teacher probs as
    weights), which carries dense supervision from tokens the student did
    NOT sample at roughly k times the training-token volume."""

    topk: int = Field(default=8, ge=1, le=64)
    """How many teacher candidates per position under `loss = "topk_ce"`
    (ignored by `importance_sampling` and `ppo`). Training volume scales
    linearly with it (k replicated cross_entropy datums per source datum)."""

    advantage_clip: Annotated[float, Field(gt=0)] | None = None
    """Symmetric bound on each per-token advantage,
    `clip(teacher_lp - student_lp, +-advantage_clip)`, applied before any
    centering.

    None (the default) trains the RAW gap, which is the OpenClaw-RL / Slime
    form and what `loss = "ppo"` expects: nothing bounds one token's
    magnitude, and the PPO ratio clip is the regularizer instead. A positive
    value caps outliers (a token the teacher is far more confident about
    than the student can otherwise dominate its batch) at the cost of
    biasing the reverse-KL estimate. `clip_fraction` in the metrics row
    reports how often the bound bit, and is 0.0 whenever clipping is off.
    Ignored by `topk_ce`, which builds no advantages."""

    center_advantages: bool = False
    """Whether to subtract the batch mean over all loss tokens from every
    loss token (after clipping), forcing the batch-mean advantage to zero.

    False (the default) trains the raw uncentered gap: a token's sign says
    whether the teacher liked it more than the student did, not whether it
    beat the batch average, and `advantage_mean` in the metrics row reads
    the objective itself (the mean teacher-minus-student gap) rather than a
    trivial 0. True restores the variance-reduced baseline form, which
    removes the shared offset but pushes DOWN every below-average token even
    when the teacher scored it above the student. Ignored by `topk_ce`.

    Uncentered is the unbiased reverse-KL estimator, and on-policy its mean is
    `-KL(student||teacher) <= 0`, so the average sampled token is pushed down
    and only tokens the teacher likes better than the student are pushed up.
    That is the correct gradient, and also why the mean is worth watching: a
    mean that stops moving toward 0 means the student stopped closing."""

    max_datum_tokens: int = Field(default=65536, ge=1)
    sampler_refresh_every: int = Field(default=1, ge=1)
    save_state_every: int = Field(default=8, ge=1)
    trial_concurrency: int = Field(default=8, ge=1)

    log_sample_rollouts: int = Field(default=2, ge=0)
    """How many sample episodes each batch renders to human-readable text:
    after every training step, the warmup collection, and each eval batch,
    the first N span-bearing trials are decoded WITH the chat template's
    special tokens and written to the run dir's `samples/` files plus the
    tracker's samples table (see `wmh.distill.samples`). 0 disables."""


class SamplingConfig(BaseModel):
    """Student sampling parameters used during rollouts.

    Both values are pinned into the harness document's param surfaces
    (`param:temperature`, `param:max-output-tokens`), which is how the pi
    runtimes stamp them onto every worker request.
    """

    model_config = ConfigDict(extra="forbid")

    temperature: float = Field(default=1.0, ge=0, le=2)
    """Rollout sampling temperature. 1.0 (the default) keeps the sampler's
    issued logprobs directly comparable to the teacher's untempered
    compute_logprobs values; any other value biases the reverse-KL advantages
    and is warned about at run start."""

    max_tokens: int = Field(default=8192, ge=1)
    """Per-completion output cap for every rollout request."""


class OffPolicyConfig(BaseModel):
    """Off-policy distillation: hard-target CE on the teacher's own trajectories.

    A primary training objective, not a bootstrap. The teacher runs the same
    terminus-2 rollouts on the TRAIN tasks (or another run's recorded
    collection is loaded through `trajectories_from`), the kept trials merge
    into cross_entropy datums through the same prefix merge on-policy
    distillation uses, and the student trains `epochs` passes over that datum
    set at `minibatch_datums` datums per optimizer step. `epochs = 0` (the
    default) disables the mode.

    The phase is resumable at datum granularity: every `checkpoint_every`
    optimizer steps it saves training state and persists a cursor
    (`offpolicy-cursor.json`), so an interrupted run continues from the next
    minibatch instead of paying for the whole schedule again.

    This supersedes the legacy `[warmup]` section, which is the same machinery
    pinned to one full-batch pass per step and no cursor. Setting both is
    rejected (`DistillConfig._check_offpolicy_supersedes_warmup`).
    """

    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(default=0, ge=0)
    """Passes over the kept teacher-trajectory datum set; 0 disables the mode."""

    minibatch_datums: int = Field(default=0, ge=0)
    """Datums per optimizer step. 0 (the default) trains each epoch as ONE
    full-batch forward_backward plus one optim_step, which is exactly what the
    legacy warmup phase did. A positive value splits each epoch into
    ceil(datums / minibatch_datums) optimizer steps in plan order; the last
    minibatch of an epoch may be short, and no datum is ever dropped or
    truncated to make the split even."""

    learning_rate: Annotated[float, Field(gt=0)] | None = None
    """Off-policy optimizer LR; None uses `train.learning_rate`."""

    rollouts_per_task: int = Field(default=1, ge=1)
    """Teacher attempts per train task when collecting the trajectory corpus."""

    keep: Literal["passed", "all"] = "passed"
    """Which teacher trials feed the datum set: reward-passing only, or all."""

    shuffle_seed: int | None = None
    """Seed for the per-epoch datum shuffle; None (the default) trains every
    epoch in build order. The order is a pure function of (seed, epoch), so a
    resumed session replays exactly the order the interrupted one planned."""

    checkpoint_every: int = Field(default=1, ge=1)
    """Save training state and rewrite the resume cursor every N optimizer
    steps. 1 (the default) makes a resume exact; a larger value trades at most
    N-1 repeated minibatches for fewer save_state round trips."""

    trajectories_from: str | None = None
    """Path to another run dir whose teacher COLLECTION completed: this run
    loads that run's `warmup-trials.json` manifest (the corpus file both CE
    phases share) instead of collecting teacher rollouts here. The manifest's
    teacher must match this run's teacher, the `keep` filter applies at load
    time, and loading charges no meter because the source run paid for it."""


class WarmupConfig(BaseModel):
    """Supervised warmup on the teacher's own pi trajectories before OPD steps.

    Superseded by `[offpolicy]`, which is the same cross_entropy machinery as a
    first-class mode (epochs, minibatches, a resumable datum cursor); this
    section stays for the runs already configured against it and is pinned to
    its historical behavior: one full-batch pass per step, no cursor, re-run
    whole on resume.

    The remedy for a student that samples only failing trajectories (on-policy
    distillation then matches the teacher on failures): before the OPD step
    loop, the teacher runs the same terminus-2 rollouts on the TRAIN tasks, its kept trials
    become cross_entropy SFT datums via the same prefix merge, and the student
    trains `steps` full-batch passes over them. 0 steps (the default) disables
    the phase entirely.
    """

    model_config = ConfigDict(extra="forbid")

    steps: int = Field(default=0, ge=0)
    """Full-batch SFT passes over the kept teacher trajectories; 0 disables."""

    rollouts_per_task: int = Field(default=1, ge=1)
    """Teacher attempts per train task when collecting warmup trajectories."""

    keep: Literal["passed", "all"] = "passed"
    """Which teacher trials feed the SFT set: reward-passing only, or all."""

    learning_rate: Annotated[float, Field(gt=0)] | None = None
    """Warmup optimizer LR; None uses `train.learning_rate`."""

    trajectories_from: str | None = None
    """Path to another run dir whose warmup COLLECTION completed: the warmup
    phase loads that run's `warmup-trials.json` manifest instead of collecting
    teacher rollouts here. The manifest's teacher must match this run's
    teacher, the `keep` filter applies at load time (so it may differ from the
    source run's), and loading charges no meter (the source run paid for the
    collection); the CE training passes still run per this run's config."""


class EvalConfig(BaseModel):
    """Held-out evaluation schedule plus cross-run baseline reuse."""

    model_config = ConfigDict(extra="forbid")

    every: int = Field(default=0, ge=0)
    """Evaluate every N train steps; 0 (the default) means final eval only."""

    tasks: int = Field(default=12, ge=1)
    k: int = Field(default=1, ge=1)

    teacher_baseline_from: str | None = None
    """Path to a prior run's `evals/baseline-teacher.json` to reuse instead of
    re-running the teacher-in-harness holdout baseline (the teacher's solve
    rate is a property of the teacher, not of the training run). The report
    must cover exactly this run's holdout task ids, carry at least `gate.k`
    attempts, and name the same teacher model; it is copied into this run's
    `evals/` with a provenance `source` note."""

    student_baseline_from: str | None = None
    """Path to a prior run's `evals/baseline-student-before.json` to reuse
    instead of re-running the pre-training student baseline (parallel runs
    from the same base model share it). Validated like
    `teacher_baseline_from`, except the model check is on the report's
    recorded `base_model` field: the student's provider model is a per-run
    sampler path and never matches across runs."""


class GateConfig(BaseModel):
    """Promotion gate for the distilled student."""

    model_config = ConfigDict(extra="forbid")

    k: int = Field(default=3, ge=1)
    min_teacher_fraction: float = Field(default=0.7, gt=0, le=1)
    require_no_regression: bool = True


CACHED_PREFILL_FRACTION = 0.2
"""Default cached-prefill price as a fraction of the full prefill price.

Tinker bills a request's verbatim repeated prompt prefix at 20% of the full
prefill price (the ratio its console lists, e.g. Ultra 0.498 vs 2.49
USD/Mtok). An explicit `*_cached_prefill` field overrides this derivation.
"""


def _effective_cached(explicit: float | None, full_prefill: float | None) -> float | None:
    """The cached-prefill rate charged: the explicit override, else 20% of the full rate."""
    if explicit is not None:
        return explicit
    if full_prefill is not None:
        return full_prefill * CACHED_PREFILL_FRACTION
    return None


class PricingConfig(BaseModel):
    """Per-model per-meter prices, USD per million tokens (all optional).

    Tinker bills prefill PER REQUEST over each call's full prompt: every
    agent turn re-bills its whole context, with the verbatim repeated prefix
    billed at a discounted cached rate. The `*_cached_prefill` fields carry
    that cached rate; when left unset they default to
    `CACHED_PREFILL_FRACTION` (20%) of the corresponding full prefill price
    whenever that price is set, and stay unpriced otherwise. `teacher_sample`
    prices the tokens teacher-in-harness episodes (warmup collection and the
    gate's teacher baseline) SAMPLE, which bill at the sampling rate
    (~2.5x prefill on the live price list), not the prefill rate.
    """

    model_config = ConfigDict(extra="forbid")

    student_prefill: Annotated[float, Field(ge=0)] | None = None
    student_cached_prefill: Annotated[float, Field(ge=0)] | None = None
    """Student cached-prefill rate; None means 20% of student_prefill when set."""

    student_sample: Annotated[float, Field(ge=0)] | None = None
    student_train: Annotated[float, Field(ge=0)] | None = None
    teacher_prefill: Annotated[float, Field(ge=0)] | None = None
    teacher_cached_prefill: Annotated[float, Field(ge=0)] | None = None
    """Teacher cached-prefill rate; None means 20% of teacher_prefill when set."""

    teacher_sample: Annotated[float, Field(ge=0)] | None = None
    """Sampling rate for teacher-in-harness episodes (warmup, gate baseline)."""

    @property
    def effective_student_cached_prefill(self) -> float | None:
        """The student cached-prefill price actually charged (see class docstring)."""
        return _effective_cached(self.student_cached_prefill, self.student_prefill)

    @property
    def effective_teacher_cached_prefill(self) -> float | None:
        """The teacher cached-prefill price actually charged (see class docstring)."""
        return _effective_cached(self.teacher_cached_prefill, self.teacher_prefill)

    def is_complete(self) -> bool:
        """Whether every meter has a price, so run cost can be fully accounted.

        The cached-prefill meters count as priced through their derived 20%
        defaults whenever the corresponding full prefill price is set (which
        this requires), so completeness needs exactly the four full prices
        plus `teacher_sample`.
        """
        return (
            self.student_prefill is not None
            and self.student_sample is not None
            and self.student_train is not None
            and self.teacher_prefill is not None
            and self.teacher_sample is not None
        )


class BudgetConfig(BaseModel):
    """Optional hard USD budget for the whole run."""

    model_config = ConfigDict(extra="forbid")

    max_usd: Annotated[float, Field(gt=0)] | None = None


PROBE_BASELINE_ENTROPY_NATS = 0.181
"""Batch-pooled entropy proxy measured on healthy, UNTRAINED Super-120B weights.

Pooled over the 356,122 sampled tokens of a live 48-episode TerminalBench-2
probe at `sampling.temperature = 0.7`
(`.wmh/distill-runs/probe-scaffold/tokens/step-0000/*.jsonl`, 47 episodes that
recorded spans). Per-episode: mean 0.200, p50 0.184, p10 0.135, min 0.082.
Recorded here as documentation for the `[tripwire]` defaults; nothing reads it
as a threshold, because each run measures its own baseline (see
`TripwireConfig`).
"""

PROBE_BASELINE_EPISODE_TOKENS = 7577
"""Batch-pooled sampled tokens per episode on the same probe.

Per-episode: p50 5,543, p10 2,242, p90 16,932, min 349, max 30,869. The 88x
spread between the shortest and longest healthy episode is why the tripwires
pool over the batch and why the length fractions are as loose as they are.
"""


class TripwireConfig(BaseModel):
    """Degeneration tripwires on the student's own sampled tokens.

    Every threshold is a FRACTION of the baseline this run measures at its own
    first training step, never an absolute number. Two sibling cross-tokenizer
    lanes died of a degeneration no KL curve shows (KL can fall while the
    policy collapses): one had generation length collapse 50x (2,866 to about
    50 tokens) while task accuracy fell 0.813 to 0.596, the other reports the
    mirror pathology, "pure KL gives EOS no gradient, so the student never
    learns to stop". `entropy_per_token` and `mean_generation_tokens` on every
    metrics row are what make both visible here.

    Why fractions and not the sibling lane's absolute "entropy below 0.2 nats
    means collapse": our own healthy, untrained baseline is
    `PROBE_BASELINE_ENTROPY_NATS` = 0.181 nats/token, so that threshold would
    fire at step 0, before a single gradient step. Terminal-command tokens are
    far more predictable than the math reasoning the sibling measured, and
    sampling at temperature 0.7 biases a sampled-token entropy estimate
    downward on its own (see `wmh.distill.tripwire.policy_health`). A tripwire
    that always fires gets muted, which is strictly worse than no tripwire, so
    do not replace any fraction below with an absolute nats or token count.

    Both bounds are one-sided, on the DOWNSIDE. The runaway direction of the
    same pathology (a student that never learns to stop) shows up as
    `mean_generation_tokens` rising instead, and the metrics row already carries
    its sharper signals: `stop_reason_counts` (`max_turns`, `output_truncated`)
    and `scaffold_loss_rate`. Nothing here aborts on it, because an episode cap
    bounds it already.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    """Whether a breach may warn or abort the run.

    False still computes both metrics, still captures and persists the
    baseline, and still records the baseline and the ratio on every metrics row
    (all of that is free, from data the step already holds); it only silences
    the warning and the abort."""

    entropy_warn_frac: float = Field(default=0.5, gt=0.0, le=1.0)
    """Warn when the batch-pooled entropy falls to this fraction of baseline.

    Half the measured 0.181 nats/token is about 0.091, which is below the
    LOWEST single healthy episode on the probe (0.082 is the per-episode min,
    and the pooled batch statistic is far tighter than any single episode)."""

    entropy_kill_frac: float = Field(default=0.3, gt=0.0, le=1.0)
    """Abort (after `kill_consecutive_steps`) at this fraction of baseline.

    About 0.054 nats/token against the measured baseline. The sibling lane's
    collapsing steps read 0.06, 0.05 and 0.13 nats against a pre-registered
    floor of 0.2, i.e. ratios of roughly 0.25 to 0.65 of their own threshold,
    so a collapse of that severity trips this while ordinary drift does not."""

    length_warn_frac: float = Field(default=0.5, gt=0.0, le=1.0)
    """Warn when the batch's mean sampled tokens per episode falls to this
    fraction of baseline (about 3,789 tokens against the measured 7,577)."""

    length_kill_frac: float = Field(default=0.25, gt=0.0, le=1.0)
    """Abort (after `kill_consecutive_steps`) at this fraction of baseline.

    A quarter of the measured 7,577 is about 1,894 tokens, still under the
    probe's per-episode p10 of 2,242: the whole batch has to average shorter
    than nine out of ten healthy episodes. The sibling's 50x collapse (2,866 to
    about 50 tokens) is a ratio of 0.017, two orders of magnitude past this."""

    kill_consecutive_steps: int = Field(default=2, ge=1)
    """Consecutive kill-level steps tolerated before the run aborts.

    Same reasoning as `MAX_CONSECUTIVE_EMPTY_STEPS`: one batch is a small task
    draw and can be transiently short or flat, two in a row is a trend. A step
    that is not at kill level resets the streak."""

    @model_validator(mode="after")
    def _check_kill_below_warn(self) -> TripwireConfig:
        """Keep every kill fraction at or under its warn fraction.

        Returns:
            This config, unchanged, when each kill threshold is reachable only
            through its warn threshold.

        Raises:
            ValueError: If a kill fraction sits above its warn fraction, which
                would abort the run without ever having warned about it.
        """
        pairs = (
            ("entropy", self.entropy_kill_frac, self.entropy_warn_frac),
            ("length", self.length_kill_frac, self.length_warn_frac),
        )
        for metric, kill, warn in pairs:
            if kill > warn:
                raise ValueError(
                    f"tripwire.{metric}_kill_frac ({kill}) must be <= "
                    f"tripwire.{metric}_warn_frac ({warn}), or the run would abort at a "
                    "ratio it never warned about first"
                )
        return self


class WandbConfig(BaseModel):
    """Optional Weights & Biases run tracking (off by default).

    Enabling it requires the wandb SDK (`uv sync --extra distill`) and
    credentials (WANDB_API_KEY or a prior `wandb login`); both are checked
    before the run spends anything (see `wmh.distill.tracking`).
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    project: str = "wmh-distill"
    entity: str | None = None
    run_name: str | None = None
    """The wandb run name; None derives one from the agent name and run dir."""

    tags: list[str] = Field(default_factory=list)


class DistillConfig(BaseModel):
    """Top-level configuration for one distillation run.

    The student, teacher, and harbor sections are required (each carries a
    required field); every other section has complete defaults and may be
    omitted from the TOML file.
    """

    model_config = ConfigDict(extra="forbid")

    student: StudentConfig
    teacher: TeacherConfig
    harbor: HarborConfig
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    offpolicy: OffPolicyConfig = Field(default_factory=OffPolicyConfig)
    warmup: WarmupConfig = Field(default_factory=WarmupConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    gate: GateConfig = Field(default_factory=GateConfig)
    pricing: PricingConfig = Field(default_factory=PricingConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    tripwire: TripwireConfig = Field(default_factory=TripwireConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    @model_validator(mode="after")
    def _check_cross_tokenizer_loss(self) -> DistillConfig:
        """Reject the losses the cross-tokenizer (chunk-aligned) path cannot express.

        Returns:
            This config, unchanged, when the loss suits the alignment.

        Raises:
            ValueError: If chunk alignment is paired with `topk_ce`.
        """
        if self.teacher.alignment == "chunk" and self.train.loss == "topk_ce":
            raise ValueError(
                'train.loss = "topk_ce" is not supported with teacher.alignment = '
                '"chunk": top-k CE trains the student on the teacher\'s candidate '
                "token ids as targets, and those ids index the TEACHER's vocabulary, "
                "which is a different vocabulary from the student's under chunk "
                "alignment (they name different text); use loss = "
                '"importance_sampling" or "ppo", which only need the teacher\'s '
                "total logprob over each chunk of the student's own tokens"
            )
        return self

    @model_validator(mode="after")
    def _check_offpolicy_supersedes_warmup(self) -> DistillConfig:
        """Reject running both cross_entropy phases in the same run.

        `[offpolicy]` and `[warmup]` train the identical objective on the
        identical corpus (the teacher's own trajectories, hard-target CE); the
        only differences are the batch shape and the resume story. Running both
        would collect the teacher twice and train the same data twice under two
        different schedules, which no run wants and no metrics row explains.

        Returns:
            This config, unchanged, when at most one CE phase is enabled.

        Raises:
            ValueError: If both phases are enabled.
        """
        if self.offpolicy.epochs > 0 and self.warmup.steps > 0:
            raise ValueError(
                f"both cross_entropy phases are enabled (offpolicy.epochs = "
                f"{self.offpolicy.epochs}, warmup.steps = {self.warmup.steps}), and they "
                "train the same objective on the same teacher trajectories: [offpolicy] "
                "supersedes [warmup] (same CE loss, plus epochs, minibatches, and a "
                "resumable datum cursor). Keep [offpolicy] and set warmup.steps = 0, or "
                "drop the [offpolicy] section"
            )
        return self

    @model_validator(mode="after")
    def _check_renderer_models(self) -> DistillConfig:
        """Reject a `rollout.renderers` key naming a model this run never samples.

        The value side is checked where it is declared (`RolloutConfig`); the KEY
        is the more dangerous typo, because a key that matches nothing is not an
        error anywhere downstream. The lookup simply misses, the model falls back
        to its auto-discovered renderer, and the run dies on trial 1 with the
        failure this setting existed to prevent.

        Returns:
            This config, unchanged, when every key names a sampled base model.

        Raises:
            ValueError: If a key is neither `student.base_model` nor `teacher.model`.
        """
        sampled = {self.student.base_model, self.teacher.model}
        unknown = sorted(set(self.rollout.renderers) - sampled)
        if unknown:
            raise ValueError(
                f"rollout.renderers names the model(s) {unknown} that this run never samples; "
                "the keys are base model names and must be student.base_model "
                f"({self.student.base_model!r}) or teacher.model ({self.teacher.model!r}). An "
                "unmatched key is silently ignored, and the model it meant to fix would fall "
                "back to its auto-discovered renderer"
            )
        return self


def load_distill_config(path: Path) -> DistillConfig:
    """Load and validate a distillation run config from a TOML file.

    Args:
        path: Path to the per-run TOML file.

    Returns:
        The validated DistillConfig.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid TOML or fails validation; the
            message names the file and each failing field.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"distill config not found: {path}") from exc
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"invalid distill config {path}: not valid TOML ({exc})") from exc
    try:
        return DistillConfig.model_validate(data)
    except ValidationError as exc:
        details = "; ".join(
            "{field}: {msg}".format(
                field=".".join(str(part) for part in err["loc"]) or "(top level)",
                msg=err["msg"],
            )
            for err in exc.errors()
        )
        raise ValueError(f"invalid distill config {path}: {details}") from exc


def snapshot_toml(cfg: DistillConfig) -> str:
    """Render a validated config back to TOML for run-dir snapshotting.

    Unset optional fields (None) are omitted; parsing the result back yields
    an identical DistillConfig.

    Args:
        cfg: The config to snapshot.

    Returns:
        A valid TOML document string.
    """
    data = cfg.model_dump(mode="json", exclude_none=True)
    return tomli_w.dumps(data)
