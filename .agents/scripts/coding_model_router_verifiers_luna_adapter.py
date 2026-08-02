"""Adapt the pinned verifiers chat dialect for Luna token limits."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger("coding-router-verifiers-luna-adapter")

UPSTREAM_COMMIT = "f6e420b9908ae14d625f079881f13c15011ee1c9"
UPSTREAM_SHA256 = "47fa2daa2e4dd2c9c1d5054a21896835e2c747d91cece988d3a1d88358abfcbc"
FUNCTION_SIGNATURE = (
    "    def apply_overrides(self, body: dict, model: str, "
    "sampling: SamplingConfig) -> dict:\n"
)
UPSTREAM_SOURCE = FUNCTION_SIGNATURE + '''\
        # Preserve the program's native fields, overlaying only what the eval owns: the model and
        # the sampling knobs it set (later keys win, so the eval's override the program's).
        return {**body, "model": model, **sampling.model_dump(exclude_none=True)}
'''
ADAPTED_SOURCE = FUNCTION_SIGNATURE + '''\
        # Preserve the program's native fields, overlaying only what the eval owns: the model and
        # the sampling knobs it set (later keys win, so the eval's override the program's).
        overrides = sampling.model_dump(exclude_none=True)
        if model == "gpt-5.6-luna":
            body = dict(body)
            native_max_tokens = body.pop("max_tokens", None)
            max_tokens = overrides.pop("max_tokens", native_max_tokens)
            if max_tokens is not None:
                overrides["max_completion_tokens"] = max_tokens
        return {**body, "model": model, **overrides}
'''


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def adapt_source(source: str) -> str:
    """Rewrite only Luna's Chat Completions token-limit field."""
    if source.count(UPSTREAM_SOURCE) != 1:
        raise ValueError("pinned ChatDialect override block is absent or ambiguous")
    return source.replace(UPSTREAM_SOURCE, ADAPTED_SOURCE, 1)


def adapt_path(chat_dialect_source: Path) -> dict[str, object]:
    """Verify and adapt the pinned chat dialect source."""
    original = chat_dialect_source.read_bytes()
    original_sha256 = _sha256_bytes(original)
    if original_sha256 != UPSTREAM_SHA256:
        raise ValueError(
            "installed ChatDialect source hash differs from the frozen verifiers commit: "
            f"{original_sha256}"
        )
    adapted = adapt_source(original.decode("utf-8")).encode()
    chat_dialect_source.write_bytes(adapted)
    return {
        "adapter": "gpt-5.6-luna-max-completion-tokens-v1",
        "upstream_commit": UPSTREAM_COMMIT,
        "original_sha256": original_sha256,
        "adapted_sha256": _sha256_bytes(adapted),
        "chat_dialect_source": str(chat_dialect_source),
        "scientific_fields_changed": False,
        "wire_mapping": "max_tokens -> max_completion_tokens for gpt-5.6-luna",
    }


def main() -> None:
    """Apply the compatibility adapter and persist its report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat-dialect-source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = adapt_path(args.chat_dialect_source)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info(
        "adapted Luna chat token field original=%s adapted=%s",
        report["original_sha256"],
        report["adapted_sha256"],
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
