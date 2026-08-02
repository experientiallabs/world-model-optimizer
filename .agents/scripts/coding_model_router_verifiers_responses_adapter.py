"""Adapt the pinned mini-swe-agent harness to its Responses model class."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger("coding-router-verifiers-responses-adapter")

UPSTREAM_COMMIT = "f6e420b9908ae14d625f079881f13c15011ee1c9"
UPSTREAM_SHA256 = "5c898dbf5fb3eb350290e193f90341ea80da705e2ea506cc5f37450c86314a78"
UPSTREAM_MODEL_CLASS_LINE = '            "litellm",'
ADAPTED_MODEL_CLASS_LINE = '            "litellm_response",'


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def adapt_source(source: str) -> str:
    """Select mini-swe-agent's pinned Responses model class."""
    if source.count(UPSTREAM_MODEL_CLASS_LINE) != 1:
        raise ValueError("pinned mini-swe-agent model-class line is absent or ambiguous")
    return source.replace(
        UPSTREAM_MODEL_CLASS_LINE, ADAPTED_MODEL_CLASS_LINE, 1
    )


def adapt_path(harness_source: Path) -> dict[str, object]:
    """Verify and adapt the pinned mini-swe-agent harness source."""
    original = harness_source.read_bytes()
    original_sha256 = _sha256_bytes(original)
    if original_sha256 != UPSTREAM_SHA256:
        raise ValueError(
            "installed mini-swe-agent harness differs from the frozen verifiers commit: "
            f"{original_sha256}"
        )
    adapted = adapt_source(original.decode("utf-8")).encode()
    harness_source.write_bytes(adapted)
    return {
        "adapter": "mini-swe-agent-litellm-responses-v1",
        "upstream_commit": UPSTREAM_COMMIT,
        "original_sha256": original_sha256,
        "adapted_sha256": _sha256_bytes(adapted),
        "harness_source": str(harness_source),
        "scientific_fields_changed": False,
        "model_class_mapping": "litellm -> litellm_response",
        "mini_swe_agent_version": "2.4.5",
    }


def main() -> None:
    """Apply the Responses adapter and persist its report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness-source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = adapt_path(args.harness_source)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info(
        "adapted mini-swe-agent to Responses original=%s adapted=%s",
        report["original_sha256"],
        report["adapted_sha256"],
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
