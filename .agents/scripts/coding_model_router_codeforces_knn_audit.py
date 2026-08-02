"""Audit focused native kNN Codeforces frontier points with grouped controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from coding_model_router_codeforces_fit import load_data
from coding_model_router_codeforces_knn import (
    ARMS,
    Candidate,
    FoldValue,
    _aggregate,
    _bootstrap,
    _dominated,
    _matrix,
    _policy,
    _routes,
    _shuffled_control,
    _static,
)
from sklearn.model_selection import GroupKFold

from wmo.optimize.knn import fit_knn_policy
from wmo.optimize.policy import EmbedderSpec

logger = logging.getLogger("coding-model-router-codeforces-knn-audit")

FOCUSED = (
    Candidate(512, "luna-xhigh", 8, 0.9, 0.5, 0.0),
    Candidate(512, "luna-xhigh", 32, 0.95, 1.0, 0.0),
    Candidate(2_048, "luna-xhigh", 32, 0.95, 1.0, 0.01),
    Candidate(2_048, "luna-max", 32, 0.95, 1.0, 0.02),
    Candidate(2_048, "luna-xhigh", 8, 0.9, 1.0, 0.0),
)


def audit(
    corpus: Path,
    outcomes: Path,
    output: Path,
    *,
    expected_tasks: int,
    seed: int,
) -> None:
    """Replay five Pareto points and their shuffled controls out of fold."""
    data = load_data(corpus, outcomes, expected_tasks=expected_tasks)
    matrix = _matrix(data)
    indices = np.arange(len(data.task_ids))
    folds = list(GroupKFold(n_splits=5).split(indices, groups=np.asarray(data.groups)))
    by_candidate: dict[str, list[FoldValue]] = defaultdict(list)
    with tempfile.TemporaryDirectory(prefix="codeforces-knn-audit-") as directory:
        root = Path(directory)
        for fold_index, (train, test) in enumerate(folds):
            for dim in sorted({candidate.dim for candidate in FOCUSED}):
                spec = EmbedderSpec(kind="hashing", dim=dim)
                embedder = spec.build()
                base = fit_knn_policy(
                    matrix,
                    bank_path=root / f"fold-{fold_index}-hash{dim}.npz",
                    fit_ids=[data.task_ids[index] for index in train],
                    embedder=spec,
                    embed_with=embedder,
                    guard_model=ARMS[0],
                    rag_num=50,
                    rag_thres=0.95,
                    z=0.0,
                    min_pairs=8,
                    se_floor=True,
                    floor_q=0.0,
                    pick_lam=0.0,
                    fitted_from="Codeforces focused external development audit",
                )
                for candidate in FOCUSED:
                    if candidate.dim == dim:
                        by_candidate[candidate.key].append(
                            _routes(data, matrix, _policy(base, candidate), test, embedder)
                        )
        static = _static(data)
        reports: list[dict[str, Any]] = []
        for position, candidate in enumerate(FOCUSED):
            value = _aggregate(by_candidate[candidate.key])
            row: dict[str, Any] = {
                "candidate": candidate.__dict__,
                "key": candidate.key,
                "reward": value["reward"],
                "cost_usd": value["cost_usd"],
                "matched_blind_reward": value["matched_blind_reward"],
                "matched_blind_cost_usd": value["matched_blind_cost_usd"],
                "advantage": value["advantage"],
                "counts": value["counts"],
            }
            row["dominated_by_static_arms"] = _dominated(row, static)
            interval = _bootstrap(data, by_candidate[candidate.key], seed=seed + position)
            shuffled_folds = _shuffled_control(
                data,
                candidate,
                folds,
                root,
                seed=seed + 10_000 + 100 * position,
            )
            shuffled = _aggregate(shuffled_folds)
            shuffled_interval = _bootstrap(
                data,
                shuffled_folds,
                seed=seed + 20_000 + position,
            )
            gate = {
                "positive_matched_blind_advantage": float(row["advantage"]) > 0.0,
                "positive_contest_bootstrap_lower_bound": interval[0] > 0.0,
                "not_static_dominated": not row["dominated_by_static_arms"],
                "shuffled_control_failed": not (
                    float(shuffled["advantage"]) > 0.0 and shuffled_interval[0] > 0.0
                ),
            }
            gate["passed"] = all(gate.values())
            row.update(
                {
                    "contest_bootstrap_advantage_95ci": interval,
                    "shuffled_control": {
                        "reward": shuffled["reward"],
                        "cost_usd": shuffled["cost_usd"],
                        "advantage": shuffled["advantage"],
                        "contest_bootstrap_advantage_95ci": shuffled_interval,
                    },
                    "gate": gate,
                }
            )
            reports.append(row)
    report = {
        "protocol": "codeforces-native-knn-focused-audit-v1",
        "tasks": len(data.task_ids),
        "contest_groups": len(set(data.groups)),
        "static_efforts": static,
        "candidates": reports,
        "passing_candidates": [row["key"] for row in reports if row["gate"]["passed"]],
        "confirmation_authorized": False,
        "target_outcomes_used": False,
        "inputs": {
            "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
            "outcomes_sha256": hashlib.sha256(outcomes.read_bytes()).hexdigest(),
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    logger.info(
        "focused audit complete passing=%d/%d",
        len(report["passing_candidates"]),
        len(FOCUSED),
    )


def main() -> None:
    """Parse the focused audit command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()
    audit(
        args.corpus,
        args.outcomes,
        args.output,
        expected_tasks=args.expected_tasks,
        seed=args.seed,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
