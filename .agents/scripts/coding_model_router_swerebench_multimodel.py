"""Run one frozen OpenAI model across all reasoning efforts on SWE-rebench."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import coding_model_router_swerebench_execute as runner

MODELS = {
    "gpt-5.6-sol": "sol",
    "gpt-5.6-terra": "terra",
}
FROZEN_PRIOR_SPEND_USD = 1_123.9297378


def configure(model: str) -> None:
    """Bind the frozen generic runner to one new model matrix."""
    if model not in MODELS:
        raise ValueError(f"unsupported frozen model: {model}")
    slug = MODELS[model]
    runner.MODEL = model
    runner.DEFAULT_PRIOR_SPEND_USD = FROZEN_PRIOR_SPEND_USD
    runner.REMOTE_VALIDATOR = runner.REMOTE_VALIDATOR.replace(
        '"gpt-5.6-luna"', f'"{model}"'
    )
    runner.REUSED_TASKS = set()
    runner.SMOKE_ARCHIVE_SHA256 = {}
    runner.DEVELOPMENT_PHASE = runner.ExecutionPhase(
        name="development",
        protocol=f"coding-router-swerebench-{slug}-development-v1",
        corpus_sha256=runner.CORPUS_SHA256,
        remote_segment=f"model-effort-v43-{slug}",
        metadata_phase=f"model-effort-v43-{slug}-development",
        reuse_smoke=False,
        metadata_owner="coding-router-v43",
    )


def main() -> None:
    """Parse the frozen launch and execute or resume it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=100)
    args = parser.parse_args()
    configure(args.model)
    runner.execute(
        args.root,
        args.corpus,
        concurrency=args.concurrency,
        limit_tasks=None,
        phase_name="development",
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
