"""End-to-end cross-tokenizer distillation: Qwen3.5-9B student, GLM-5.2 teacher.

Single-turn math, so there is no harbor, no E2B, and no multi-turn prefix merge:
each step samples the student on a batch of problems, has GLM-5.2 score the
student's own text through its OWN tokenization, chunk-aligns the two token
sequences, and trains the resulting per-token advantages through Tinker's
`importance_sampling` loss.

The loop, per step:
  1. sample `group_size` completions per problem from the Tinker LoRA student
  2. build one TrainDatum per completion (prompt tokens masked, sampled tokens
     carrying loss), which is the TITO contract: the exact sampled ids train
  3. render the same conversation with GLM's chat template and locate the
     byte-identical content islands
  4. score the teacher's rendered token ids on Fireworks (`echo`)
  5. chunk-align, attach chunk advantages, forward_backward + optim_step

Usage:
    uv run python .agents/distill/xtoken_run.py --steps 2 --problems 4 --group 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger("xtoken-run")

SYSTEM_PROMPT = (
    "You are a careful mathematician. Solve the problem and put your final answer in \\boxed{}."
)
TEACHER_MODEL = "accounts/fireworks/models/glm-5p2"
TEACHER_TOKENIZER = "zai-org/GLM-5.2"
_CONTEXT_MARGIN = 16
"""Tokens held back from the context so the sampler cannot overrun it."""
FIREWORKS_URL = "https://api.fireworks.ai/inference/v1"


def load_train_problems(limit: int, *, integers_only: bool = True) -> list[dict[str, str]]:
    """DeepScaleR problems, optionally restricted to integer answers.

    Integer answers match AIME's format and are the only ones the grader scores
    unambiguously, so reward metrics stay trustworthy. The filter does not touch
    the training signal: the loss is reverse KL against the teacher's logprobs
    and never reads the answer.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "agentica-org/DeepScaleR-Preview-Dataset", "deepscaler.json", repo_type="dataset"
    )
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[dict[str, str]] = []
    for row in rows:
        answer = str(row.get("answer", "")).strip()
        if integers_only and not answer.lstrip("-").isdigit():
            continue
        out.append({"problem": str(row["problem"]), "answer": answer})
        if limit and len(out) >= limit:
            break
    return out


def main() -> None:  # noqa: C901 - a linear driver reads better than split phases
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--problems", type=int, default=4, help="problems per step")
    parser.add_argument("--group", type=int, default=2, help="samples per problem")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="0 (default) means the model's whole remaining context per problem: no rollout "
        "is ever truncated by our choice of budget",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--advantage-clip", type=float, default=4.0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--out", default=".wmh/xtoken-runs/smoke")
    parser.add_argument("--wandb-project", default="wmh-distill")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="save a resumable training checkpoint every N steps (0 disables)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from the run dir's checkpoint and continue at the next step",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    import tinker
    from llm_waterfall.types import ChatMessage
    from transformers import AutoTokenizer

    from wmh.distill.config import DistillConfig
    from wmh.distill.data import TrainDatum, to_tinker_datums
    from wmh.distill.rendering import build_renderer
    from wmh.distill.xtoken.chunks import attach_chunk_advantages
    from wmh.distill.xtoken.plan import build_chunk_plan
    from wmh.distill.xtoken.prompt_logprobs import PromptLogprobClient
    from wmh.distill.xtoken.teacher_render import render_for_teacher

    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        raise SystemExit("FIREWORKS_API_KEY is not set; source platform/.env.local")

    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    tracker = None
    if not args.no_wandb:
        import wandb

        tracker = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"xtoken-{run_dir.name}",
            config={
                "student": args.student,
                "teacher": TEACHER_MODEL,
                "teacher_tokenizer": TEACHER_TOKENIZER,
                "alignment": "chunk",
                "steps": args.steps,
                "problems_per_step": args.problems,
                "group_size": args.group,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "lora_rank": args.lora_rank,
                "learning_rate": args.learning_rate,
                "advantage_clip": args.advantage_clip,
                "train_set": "DeepScaleR-Preview-Dataset (integer answers)",
            },
        )
        logger.info("WANDB RUN URL: %s", tracker.url)

    logger.info("building tinker clients for %s", args.student)
    service = tinker.ServiceClient()
    training = service.create_lora_training_client(base_model=args.student, rank=args.lora_rank)
    context_limit = next(
        (
            m.max_context_length
            for m in service.get_server_capabilities().supported_models
            if m.model_name == args.student and m.max_context_length
        ),
        None,
    )
    if context_limit is None:
        raise SystemExit(
            f"could not read max_context_length for {args.student} from the Tinker catalog; "
            "pass --max-tokens explicitly"
        )
    logger.info("model context limit: %d tokens", context_limit)

    cfg = DistillConfig.model_validate(
        {
            "student": {"base_model": args.student, "lora_rank": args.lora_rank},
            "teacher": {
                "backend": "openai_compat",
                "model": TEACHER_MODEL,
                "tokenizer": TEACHER_TOKENIZER,
                "alignment": "chunk",
                "endpoint": FIREWORKS_URL,
            },
            "harbor": {"job_template": "unused-single-turn.yaml"},
            "train": {
                "advantage_clip": args.advantage_clip,
                "center_advantages": True,
                "learning_rate": args.learning_rate,
            },
            "sampling": {
                "temperature": args.temperature,
                # The resolved ceiling, never the 0 sentinel: the snapshot must
                # record the budget actually in force.
                "max_tokens": args.max_tokens or context_limit,
            },
        }
    )

    student_tokenizer = AutoTokenizer.from_pretrained(args.student)
    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_TOKENIZER)
    rendering = build_renderer(args.student, student_tokenizer)
    teacher = PromptLogprobClient(FIREWORKS_URL, TEACHER_MODEL, api_key=key, dialect="echo")
    verify = teacher.verify()
    logger.info("teacher verify: ok=%s %s", verify.ok, (verify.detail or "")[:80])
    if not verify.ok:
        raise SystemExit("teacher endpoint did not verify; aborting before spending")

    checkpoint_path = run_dir / "checkpoint.json"
    start_step = 0
    if args.resume and checkpoint_path.exists():
        saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        training.load_state(saved["state_path"]).result()
        start_step = int(saved["next_step"])
        logger.info(
            "resumed from %s at step %d (trained weights restored)",
            saved["state_path"],
            start_step,
        )
    elif args.resume:
        logger.warning("--resume given but %s does not exist; starting fresh", checkpoint_path)

    problems = load_train_problems(args.problems * args.steps)
    logger.info("loaded %d DeepScaleR problems (integer answers)", len(problems))

    stop = rendering.stop_sequences
    stop_strings = stop if stop and isinstance(stop[0], str) else None

    def budget_for(prompt_length: int) -> int:
        """Output budget for one prompt: the whole remaining context.

        Truncation is never an acceptable outcome (it trains the student to
        imitate unfinished reasoning AND makes every accuracy number a floor),
        so the default budget is everything the context allows rather than a
        round number. `--max-tokens` only ever LOWERS it.
        """
        room = context_limit - prompt_length - _CONTEXT_MARGIN
        if room < 1:
            raise SystemExit(
                f"prompt of {prompt_length} tokens leaves no room in a {context_limit}-token "
                "context; drop the problem or use a longer-context student"
            )
        return min(args.max_tokens, room) if args.max_tokens else room

    for step in range(start_step, args.steps):
        step_started = time.time()
        batch = problems[step * args.problems : (step + 1) * args.problems]
        if not batch:
            logger.warning("no problems left for step %d", step)
            break
        sampler_path = training.save_weights_for_sampler(name=f"step-{step:04d}").result().path
        sampler = service.create_sampling_client(model_path=sampler_path)
        logger.info("step %d: sampling %d x %d from %s", step, len(batch), args.group, sampler_path)

        def sample_one(
            item: tuple[int, int, dict[str, str]],
            sampler: tinker.SamplingClient = sampler,
        ) -> dict[str, object] | None:
            index, attempt, row = item
            messages = [
                ChatMessage(role="system", content=SYSTEM_PROMPT),
                ChatMessage(role="user", content=row["problem"]),
            ]
            prompt_ids = rendering.build_generation_prompt(messages)
            allowed = budget_for(len(prompt_ids))
            response = sampler.sample(
                prompt=tinker.ModelInput.from_ints(prompt_ids),
                num_samples=1,
                sampling_params=tinker.SamplingParams(
                    max_tokens=allowed,
                    temperature=args.temperature,
                    stop=stop_strings,
                ),
            ).result()
            sequence = response.sequences[0]
            if len(sequence.tokens) >= allowed:
                # Hit the ceiling of the whole context: the episode is
                # unfinished, so it is dropped rather than trained on.
                logger.warning(
                    "problem %d attempt %d used its whole %d-token budget without "
                    "finishing; DROPPING it from training rather than teaching truncated "
                    "reasoning",
                    index,
                    attempt,
                    allowed,
                )
                return None
            if not sequence.logprobs:
                logger.warning(
                    "problem %d attempt %d returned no logprobs; dropping", index, attempt
                )
                return None
            return {
                "index": index,
                "attempt": attempt,
                "problem": row["problem"],
                "prompt_ids": list(prompt_ids),
                "sampled_ids": list(sequence.tokens),
                "sampled_logprobs": [float(x) for x in sequence.logprobs],
            }

        work = [
            (index, attempt, row)
            for index, row in enumerate(batch)
            for attempt in range(args.group)
        ]
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            rollouts = [item for item in pool.map(sample_one, work) if item is not None]
        dropped = len(work) - len(rollouts)
        logger.info(
            "step %d: %d rollout(s) sampled, %d dropped (unfinished or logprob-less)",
            step,
            len(rollouts),
            dropped,
        )

        datums: list[TrainDatum] = []
        plans = []
        rows = []
        coverage_num = 0
        coverage_den = 0
        for rollout in rollouts:
            prompt_ids = list(rollout["prompt_ids"])  # type: ignore[arg-type]
            sampled_ids = list(rollout["sampled_ids"])  # type: ignore[arg-type]
            sampled_lp = list(rollout["sampled_logprobs"])  # type: ignore[arg-type]
            if not sampled_ids:
                continue
            datum = TrainDatum(
                trial_name=f"p{rollout['index']}-a{rollout['attempt']}",
                fragment_index=0,
                model_input_tokens=prompt_ids + sampled_ids,
                loss_mask=[0.0] * len(prompt_ids) + [1.0] * len(sampled_ids),
                sampled_logprobs=[0.0] * len(prompt_ids) + sampled_lp,
            )
            # The conversation the teacher must render: the student's own text.
            student_text = rendering.decode(sampled_ids)
            messages = [
                ChatMessage(role="system", content=SYSTEM_PROMPT),
                ChatMessage(role="user", content=str(rollout["problem"])),
                ChatMessage(role="assistant", content=student_text),
            ]
            render = render_for_teacher(teacher_tokenizer, messages)
            if not render.islands:
                logger.warning("no islands for %s; skipping", datum.trial_name)
                continue
            plan = build_chunk_plan(datum, render, student_tokenizer, teacher_tokenizer)
            if not plan.chunks:
                logger.warning("no chunks for %s; skipping", datum.trial_name)
                continue
            try:
                row = teacher.score(list(render.token_ids))
            except Exception as exc:  # noqa: BLE001 - one bad trajectory must not kill the step
                logger.warning("teacher scoring failed for %s: %s", datum.trial_name, exc)
                continue
            datums.append(datum)
            plans.append(plan)
            rows.append(row)
            coverage_num += plan.scored_student_tokens
            coverage_den += datum.loss_token_count

        if not datums:
            logger.error("step %d produced no scoreable datums; skipping optimizer step", step)
            continue

        trained, stats = attach_chunk_advantages(datums, plans, rows, cfg)
        logger.info(
            "step %d: %d datum(s) trained | coverage %.1f%% | chunks %d | chunk reverse-KL %s",
            step,
            stats.datums,
            100 * stats.coverage_rate,
            stats.chunks,
            "n/a" if stats.chunk_reverse_kl is None else f"{stats.chunk_reverse_kl:.4f}",
        )
        if not trained:
            logger.error("step %d: nothing survived advantage attachment", step)
            continue

        wire = to_tinker_datums(trained)
        fwd = training.forward_backward(wire, loss_fn="importance_sampling").result()
        opt = training.optim_step(tinker.AdamParams(learning_rate=args.learning_rate)).result()
        if args.save_every and (step + 1) % args.save_every == 0:
            state_path = training.save_state(name=f"step-{step:04d}").result().path
            # Written last and atomically: a crash mid-write must not leave a
            # checkpoint pointing at a step whose optimizer step did not land.
            temporary = checkpoint_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps({"state_path": state_path, "next_step": step + 1}),
                encoding="utf-8",
            )
            temporary.replace(checkpoint_path)
            logger.info("checkpoint saved: %s (resume at step %d)", state_path, step + 1)

        metrics = {
            "step": step,
            "rollouts": len(rollouts),
            "dropped_rollouts": dropped,
            "datums": stats.datums,
            "chunks": stats.chunks,
            "coverage_rate": stats.coverage_rate,
            "scored_loss_tokens": stats.scored_loss_tokens,
            "unscored_loss_tokens": stats.unscored_loss_tokens,
            "chunk_reverse_kl": stats.chunk_reverse_kl,
            "advantage_mean": stats.advantage_mean,
            "advantage_std": stats.advantage_std,
            "clipped_chunks": stats.clipped_chunks,
            "mismatch_drops": stats.mismatch_drops,
            "teacher_tokens": teacher.usage(),
            "teacher_placeholder_retries": teacher.placeholder_responses(),
            "mean_sampled_tokens": sum(len(r["sampled_ids"]) for r in rollouts) / len(rollouts),  # type: ignore[arg-type]
            "seconds": round(time.time() - step_started, 1),
            "fwd_metrics": {k: float(v) for k, v in (getattr(fwd, "metrics", {}) or {}).items()},
            "opt_metrics": {k: float(v) for k, v in (getattr(opt, "metrics", {}) or {}).items()},
        }
        with metrics_path.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(metrics) + "\n")
        if tracker is not None:
            tracker.log(
                {k: v for k, v in metrics.items() if isinstance(v, (int, float))}, step=step
            )
        logger.info(
            "step %d done in %.0fs | %s",
            step,
            metrics["seconds"],
            json.dumps(
                {
                    k: metrics[k]
                    for k in (
                        "coverage_rate",
                        "chunk_reverse_kl",
                        "teacher_tokens",
                        "teacher_placeholder_retries",
                        "mean_sampled_tokens",
                    )
                }
            ),
        )

    teacher.close()
    if tracker is not None:
        logger.info("WANDB RUN URL: %s", tracker.url)
        tracker.finish()
    logger.info("run complete; metrics at %s", metrics_path)


if __name__ == "__main__":
    main()
