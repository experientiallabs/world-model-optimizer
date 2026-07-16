"""Box-side LoRA SFT on privileged-teacher episodes (charter stage 3). Runs on GPU 1.

Input: teacher_episodes.jsonl (Qwen-convention messages). Each episode explodes into per-turn
samples: prompt = chat-template rendering of (system, user, prior turns WITH think stripped —
exactly the distribution the serving scaffold produces), completion = the target assistant turn
WITH its <think> block (BENCH-B2 showed training without deliberation breaks policy checking).
Loss on completion only (TRL prompt-completion format). LoRA adapter saved small (disk is tight).

Usage (on the box, wmh-distill venv):
    CUDA_VISIBLE_DEVICES=1 python train_sft_box.py \
        --data /mnt/azureuser/wmh_distill/teacher_episodes.jsonl \
        --out /mnt/azureuser/wmh_distill/adapter_v1 [--epochs 3]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def explode_episode(record: dict) -> list[dict]:
    """One sample per assistant turn; prior assistant turns get think-blocks stripped."""
    messages = record["messages"]
    samples = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        history = []
        for prior in messages[:index]:
            content = prior["content"]
            if prior["role"] == "assistant":
                content = THINK_RE.sub("", content).strip()
            history.append({"role": prior["role"], "content": content})
        samples.append(
            {
                "history": history,
                "completion": message["content"],
                "scenario_id": record["scenario_id"],
            }
        )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-len", type=int, default=8192)
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    records = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line]
    samples = [s for r in records for s in explode_episode(r)]
    print(f"{len(records)} episodes -> {len(samples)} per-turn samples")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = []
    dropped = 0
    for sample in samples:
        prompt = tokenizer.apply_chat_template(
            sample["history"], tokenize=False, add_generation_prompt=True
        )
        completion = sample["completion"] + tokenizer.eos_token
        if len(tokenizer(prompt + completion).input_ids) > args.max_len:
            dropped += 1
            continue
        rows.append({"prompt": prompt, "completion": completion})
    print(f"dataset: {len(rows)} rows ({dropped} dropped over max-len)")
    dataset = Dataset.from_list(rows)

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        # No mid-training checkpoints: only the final adapter is ever used, and the checkpoint's
        # optimizer state (~1 GB transient, written before the old one is deleted) overflows the
        # box's nearly-full /mnt. Pure I/O change — the optimization trajectory is unaffected.
        save_strategy="no",
        bf16=True,
        max_length=args.max_len,
        gradient_checkpointing=True,
        report_to=[],
        completion_only_loss=True,
        model_init_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"},
    )
    trainer = SFTTrainer(
        model=args.model,
        args=config,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.out)
    print(f"adapter saved -> {args.out}")


if __name__ == "__main__":
    main()
