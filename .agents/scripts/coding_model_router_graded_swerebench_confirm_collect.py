"""Collect the sealed graded SWE-rebench confirmation matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import coding_model_router_graded_swerebench_collect as collector

PROTOCOL = "coding-router-graded-swerebench-confirmation-collection-v1"
EXECUTION_PROTOCOL = "coding-router-graded-swerebench-confirmation-execution-v1"
CORPUS_SHA256 = "c9443c9956e496123f396ee793efbb3368312092c4dcbd4e5e10bb77bd814f0a"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collector.PROTOCOL = PROTOCOL
    collector.EXECUTION_PROTOCOL = EXECUTION_PROTOCOL
    collector.PHASE_NAME = "confirmation"
    collector.CORPUS_SHA256 = CORPUS_SHA256
    collector.SOURCE_TASKS = 320
    collector.MIN_RETAINED_TASKS = 304
    collector.collect(args.root, args.corpus, args.output)


if __name__ == "__main__":
    main()
