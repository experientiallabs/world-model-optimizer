"""Verify the production knn fold-in (PR #259): research selector vs knn_decision, per pick.

Runs BOTH implementations on identical inputs and asserts per-scenario pick equality:

- research: r1_retrieval_ablations.route() with the adapt3 config (adaptive rag, stat guard
  z=0.5, small-sample SE floor) and the quantile novelty floor.
- production: wmo.optimize.knn.fit_knn_policy + wmo.optimize.policy.knn_decision from the
  feat/knn-policy checkout this script is run inside.

Grid: {routerbench-ours9, wm-tau-bench, wm-financebench, wm-continual-learning} x seeds 0-4
x floor_q {0.0, 0.05} x baseline {fit-chosen, pinned fable-5}. Any divergence prints the
scenario, both picks, and both reasons; the exit summary counts matches/divergences per cell.
Cached oai3l vectors are served to both sides (embed_with= for production), so embedding is
identical by construction and any divergence is selection semantics.

Usage (the kNN policy it verifies is now on main, so no separate worktree is needed):
    uv run python .agents/scripts/r1_verify_production.py
"""

from __future__ import annotations

import importlib.util
import logging
import tempfile
from pathlib import Path

import numpy as np

from wmo.optimize.knn import fit_knn_policy
from wmo.optimize.outcomes import OutcomeMatrix
from wmo.optimize.policy import EmbedderSpec, knn_decision
from wmo.research.routerbench import best_single_model, split_scenario_ids
from wmo.research.routing_corpus import routing_data

_spec = importlib.util.spec_from_file_location(
    "r1_retrieval_ablations",
    Path(__file__).with_name("r1_retrieval_ablations.py"),
)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MatrixContext = _mod.MatrixContext
RetrievalParams = _mod.RetrievalParams

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("r1.verify")

DATA = routing_data()
MATRICES = ["routerbench-ours9", "tau-bench", "financebench", "continual-learning"]


class _CachedEmbedder:
    """Serves the research run's exact vectors to the production fit, keyed by text."""

    def __init__(self, mapping: dict[str, np.ndarray]) -> None:
        self._mapping = mapping

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, self._mapping[text])) for text in texts]


def adaptive_params(bank: int, floor: float | None) -> RetrievalParams:
    rag = min(50, max(4, -(-bank // 2)))
    return RetrievalParams(
        second_route=False,
        guard="stat",
        z=0.5,
        rag_num=rag,
        min_pairs=min(8, max(3, rag // 2)),
        se_floor=True,
        distance_floor=floor,
    )


def main() -> None:
    total_match = total_diverge = 0
    for name in MATRICES:
        file_stem = name if name == "routerbench-ours9" else name
        ctx_name = name if name == "routerbench-ours9" else f"wm-{name}"
        matrix = OutcomeMatrix.load(DATA / "matrices" / f"{file_stem}_matrix.json")
        ctx = MatrixContext(matrix, ctx_name, embed="openai", embed_replies=False)
        text_to_vec = {ctx.tasks[sid]: ctx.task_vecs[sid] for sid in ctx.scenario_ids}
        embedder = _CachedEmbedder(text_to_vec)
        for seed in range(5):
            fit_ids, test_ids = split_scenario_ids(matrix, train_fraction=0.7, seed=seed)
            fit_best, _, _ = best_single_model(matrix, fit_ids=fit_ids, eval_ids=test_ids)
            for baseline in (fit_best, "fable-5"):
                if baseline not in {e.name for e in matrix.pool}:
                    continue
                for floor_q in (0.0, 0.05):
                    with tempfile.TemporaryDirectory() as tmp:
                        policy = fit_knn_policy(
                            matrix,
                            bank_path=Path(tmp) / "bank.npz",
                            fit_ids=fit_ids,
                            # Offline check: the spec is a dim-matching placeholder; queries
                            # are fed pre-embedded, so build() is never called.
                            embedder=EmbedderSpec(dim=3072),
                            embed_with=embedder,
                            guard_model=baseline,
                            floor_q=floor_q,
                        )
                    # Research side: same adaptive rule + the same floor value the
                    # production fit derived (both from fit-bank geometry only).
                    params = adaptive_params(len(fit_ids), policy.floor_sim)
                    research = _mod.route(ctx, params, fit_ids, test_ids, baseline)
                    diverged = []
                    for sid in test_ids:
                        decision = knn_decision(policy, ctx.task_vecs[sid])
                        if decision.model != research[sid]:
                            diverged.append((sid, research[sid], decision.model, decision.reason))
                    total_match += len(test_ids) - len(diverged)
                    total_diverge += len(diverged)
                    tag = (
                        f"{ctx_name} seed{seed} base={baseline} floor_q={floor_q}: "
                        f"{len(test_ids) - len(diverged)}/{len(test_ids)} match"
                    )
                    print(tag + ("" if not diverged else f"  DIVERGED {len(diverged)}"))  # noqa: T201
                    for sid, mine, theirs, reason in diverged[:5]:
                        print(f"   {sid}: research={mine} production={theirs} | {reason}")  # noqa: T201
    print(f"\nTOTAL: {total_match} match, {total_diverge} diverge")  # noqa: T201


if __name__ == "__main__":
    main()
