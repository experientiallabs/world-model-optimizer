"""Collect one selected model-effort confirmation arm."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import coding_model_router_swerebench_collect as collector

MODELS = {
    "gpt-5.6-luna": ("luna", (1.0, 0.1, 6.0)),
    "gpt-5.6-terra": ("terra", (2.5, 0.25, 15.0)),
    "gpt-5.6-sol": ("sol", (5.0, 0.5, 30.0)),
}
CONFIRMATION_CORPUS_SHA256 = (
    "9798dd1e58be0d13331d097307670dc3fc3760ad211da20e6367666523f080a7"
)


def configure(model: str, effort: str) -> str:
    """Bind collection to one exact selected confirmation arm."""
    if model not in MODELS or effort not in collector.EFFORTS:
        raise ValueError("unsupported selected confirmation arm")
    slug, prices = MODELS[model]
    arm = f"{slug}-{effort}"
    collector.EFFORTS = (effort,)
    collector.DEVELOPMENT_PHASE = collector.CollectionPhase(
        name="development",
        protocol=f"coding-router-model-effort-confirmation-{arm}-collection-v1",
        execution_protocol=f"coding-router-model-effort-confirmation-{arm}-v1",
        corpus_sha256=CONFIRMATION_CORPUS_SHA256,
        provenance=f"model-effort-v43-confirmation-{arm}",
        reuse_smoke=False,
        requires_authorization=True,
        model=model,
        arm_prefix=slug,
        prices_per_mtok=prices,
    )
    return arm


def main() -> None:
    """Parse and collect one selected arm."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--effort", choices=collector.EFFORTS, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arm = configure(args.model, args.effort)
    collector.collect(
        args.root,
        args.corpus,
        None,
        args.output,
        phase_name="development",
    )
    logging.getLogger("coding-router-model-effort-confirm-collect").info(
        "collected selected confirmation arm=%s", arm
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
