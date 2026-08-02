"""Collect one frozen non-Luna SWE-rebench model matrix."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import coding_model_router_swerebench_collect as collector

MODELS = {
    "gpt-5.6-sol": ("sol", (5.0, 0.5, 30.0)),
    "gpt-5.6-terra": ("terra", (2.5, 0.25, 15.0)),
}


def configure(model: str) -> None:
    """Bind the generic collector to one frozen model matrix."""
    if model not in MODELS:
        raise ValueError(f"unsupported frozen model: {model}")
    slug, prices = MODELS[model]
    collector.DEVELOPMENT_PHASE = collector.CollectionPhase(
        name="development",
        protocol=f"coding-router-swerebench-{slug}-development-collection-v1",
        execution_protocol=f"coding-router-swerebench-{slug}-development-v1",
        corpus_sha256=collector.CORPUS_SHA256,
        provenance=f"model-effort-v43-{slug}-development-matrix",
        reuse_smoke=False,
        requires_authorization=False,
        model=model,
        arm_prefix=slug,
        prices_per_mtok=prices,
    )


def main() -> None:
    """Parse paths and collect a complete frozen model matrix."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure(args.model)
    collector.collect(
        args.root,
        args.corpus,
        None,
        args.output,
        phase_name="development",
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
