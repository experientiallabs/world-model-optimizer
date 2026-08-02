"""Adapt pinned SWE-rebench Prime image aliases for a Docker runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger("coding-router-swerebench-docker-adapter")

UPSTREAM_COMMIT = "a90fbd708de9ab18f85b5ffc3a0bdc60825dcc84"
UPSTREAM_SHA256 = "b7dab7d2b263d6d296cc9e2e4b9b4597cc3fbba040d3036a139cf0fb4432e730"
UPSTREAM_IMAGE_LINE = '                    image=row["image_name"],'
ADAPTED_IMAGE_LINE = '                    image=_coding_router_docker_image(row["image_name"]),'
FUNCTION_ANCHOR = "\ndef repo_workdir(repo: str) -> str:\n"
FUNCTION_SOURCE = '''
def _coding_router_docker_image(image: str) -> str:
    """Invert the filtered dataset's documented Prime registry rewrite."""
    prefix = "prime/primeintellect/"
    if not image.startswith(prefix):
        raise ValueError(f"unexpected SWE-rebench image alias: {image!r}")
    suffix = image.removeprefix(prefix)
    if not suffix or suffix != suffix.strip() or suffix.startswith("/"):
        raise ValueError(f"invalid SWE-rebench image alias: {image!r}")
    return f"docker.io/swerebenchv2/{suffix}"


def repo_workdir(repo: str) -> str:
'''


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def adapt_source(source: str) -> str:
    """Return the pinned taskset source with only image resolution adapted."""
    if source.count(UPSTREAM_IMAGE_LINE) != 1:
        raise ValueError("pinned image construction line is absent or ambiguous")
    if source.count(FUNCTION_ANCHOR) != 1:
        raise ValueError("pinned function anchor is absent or ambiguous")
    adapted = source.replace(FUNCTION_ANCHOR, "\n" + FUNCTION_SOURCE, 1)
    return adapted.replace(UPSTREAM_IMAGE_LINE, ADAPTED_IMAGE_LINE, 1)


def adapt_path(taskset_source: Path) -> dict[str, object]:
    """Verify and adapt one installed copy of the pinned official taskset."""
    original = taskset_source.read_bytes()
    original_sha256 = _sha256_bytes(original)
    if original_sha256 != UPSTREAM_SHA256:
        raise ValueError(
            "installed taskset source hash differs from the frozen upstream commit: "
            f"{original_sha256}"
        )
    adapted = adapt_source(original.decode("utf-8")).encode()
    taskset_source.write_bytes(adapted)
    return {
        "adapter": "prime-alias-to-docker-hub-v1",
        "upstream_commit": UPSTREAM_COMMIT,
        "original_sha256": original_sha256,
        "adapted_sha256": _sha256_bytes(adapted),
        "taskset_source": str(taskset_source),
        "scientific_fields_changed": False,
        "image_mapping": "prime/primeintellect/<name>:<tag> -> docker.io/swerebenchv2/<name>:<tag>",
    }


def main() -> None:
    """Apply the adapter and persist a content-addressed report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--taskset-source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = adapt_path(args.taskset_source)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info(
        "adapted pinned SWE-rebench taskset original=%s adapted=%s",
        report["original_sha256"],
        report["adapted_sha256"],
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
