"""Audit the post-hoc routing bounds on the shared outcome matrices.

Reproduces the three results from the drawing-board survey (2026-07-24), from data only:

  (a) BEST-OF-N CEILING. Taking the better of the episodes a cell already holds beats the best
      single model on BOTH accuracy and cost on several corpora - e.g. tau-bench kimi-k2.6 at
      +7.1pt and 3.6x cheaper, terminal-tasks gpt-5.4-mini at accuracy parity and 10.7x cheaper.
  (b) A FREE SELECTOR REACHES ABOUT A THIRD OF IT, ONCE THE SIGN IS RIGHT. Selecting on the
      POOLED correlation's sign (prefer fewer steps) is worse than a coin flip - 35-61% correct,
      harvesting -32%..+14%. Inverting it to "prefer the episode that finished, then the one that
      did MORE work" harvests +12%..+53% at 60-79% correct, pooled 67.5% over 440 decisive cells
      (z=+7.34). No extra call, no added latency, so best-of-n does not need an LLM verifier for a
      first cut. Both components survive their controls (section b2): the finish term at 69.7% on
      165 cells (z=+5.06), and the effort term at 66.2% on 275 cells where BOTH episodes finished
      (z=+5.37) - the latter rules out max_steps truncation as the whole explanation. The finish
      term should weaken when the corpora are recaptured at a higher step cap; the effort term is
      cap-independent.
  (c) WHY THE SIGN FLIPS. `stop_reason == max_steps` correlates -0.33..-0.43 with reward pooled
      across rows, and effort correlates negatively too - but that is BETWEEN-cell difficulty
      (hard scenarios burn steps and score low). WITHIN a cell the effort sign reverses
      (tau-bench steps +0.310, n_replies +0.363). Reading the pooled number alone is what produced
      the backwards selector in (b), so this decomposition is the guard against repeating it.

CAVEAT: `finished-then-more-steps` was chosen after trying ~7 free selectors, so its exact harvest
is optimistically biased. The controls in (b2) test the MECHANISM rather than re-searching, which
is why they carry the argument; treat the harvest percentages as directional.

Usage:
    uv run .agents/scripts/audit_posthoc_bounds.py [--matrices DIR] [--top N]

Reads `*_matrix.json` from the shared routing data dir (default below); writes nothing.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.research.posthoc_bounds import (
    DEFAULT_SELECTOR,
    SELECTOR_KEYS,
    all_finished,
    corpus_bounds,
    feature_correlations,
    one_finished,
    pooled_correct_z,
    selector_bound,
)
from wmo.research.routing_corpus import routing_data

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("posthoc-bounds")

MATRICES = routing_data() / "matrices"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrices", type=Path, default=MATRICES, help="dir of *_matrix.json")
    parser.add_argument("--top", type=int, default=4, help="models per corpus in the ceiling table")
    args = parser.parse_args()

    paths = sorted(args.matrices.glob("*_matrix.json"))
    if not paths:
        raise SystemExit(f"no *_matrix.json under {args.matrices}")
    loaded = [(path.stem.removesuffix("_matrix"), OutcomeMatrix.load(path)) for path in paths]

    logger.info("=" * 100)
    logger.info("(a) ANCHORS + BEST-OF-N CEILING  (oracle-of-n = perfect verifier, upper bound)")
    logger.info("=" * 100)
    for corpus, matrix in loaded:
        bounds = corpus_bounds(matrix, corpus)
        spread = (
            f"episode spread {bounds.episode_disagreement_mean:.3f}"
            f" ({bounds.episode_disagreement_fraction:.0%} of cells disagree)"
            if bounds.episode_disagreement_mean is not None
            else "single-episode matrix (no best-of-n)"
        )
        logger.info(
            "\n%s  n=%d models=%d episodes/cell=%s",
            corpus,
            bounds.scenarios,
            bounds.models,
            bounds.episodes_per_cell,
        )
        logger.info(
            "  best-single %s: %.3f @ $%.5f | cross-model oracle %.3f (+%.3f) | %s",
            bounds.best_single,
            bounds.best_single_accuracy,
            bounds.best_single_cost_per_call,
            bounds.oracle_accuracy,
            bounds.oracle_accuracy - bounds.best_single_accuracy,
            spread,
        )
        for bound in bounds.best_of_n[: args.top]:
            if bound.beats_best_single_accuracy and bound.beats_best_single_cost:
                flag = "  <== BEATS BEST-SINGLE ON BOTH AXES"
            elif bound.beats_best_single_accuracy:
                flag = "  (accuracy win, costs more)"
            else:
                flag = ""
            logger.info(
                "    %-16s 1-shot %.3f @ $%.5f -> best-of-%d selected %.3f (oracle %.3f)"
                " @ $%.5f%s",
                bound.model,
                bound.one_shot_accuracy,
                bound.one_shot_cost_per_call,
                bound.episodes,
                bound.selected_of_n_accuracy,
                bound.oracle_of_n_accuracy,
                bound.oracle_of_n_cost_per_call,
                flag,
            )

    logger.info("\n" + "=" * 100)
    logger.info("(b) WITHIN-CELL SELECTION BY FREE FEATURES  (harvest 0% == chance)")
    logger.info("=" * 100)
    for feature, key in SELECTOR_KEYS.items():
        logger.info("\n%s", feature)
        for corpus, matrix in loaded:
            try:
                bound = selector_bound(matrix, corpus, feature, key)
            except ValueError:
                logger.info("  %-19s single-episode matrix, skipped", corpus)
                continue
            logger.info(
                "  %-19s random %.3f | selector %.3f | oracle %.3f | %3d decisive,"
                " %3.0f%% correct | harvest %+4.0f%%",
                corpus,
                bound.random_of_n,
                bound.selector_accuracy,
                bound.oracle_of_n,
                bound.decisive_cells,
                bound.correct_fraction * 100,
                bound.harvested_fraction * 100,
            )

    logger.info("\n" + "=" * 100)
    logger.info("(b2) CONFOUND CONTROLS for %s", DEFAULT_SELECTOR)
    logger.info("=" * 100)
    controls: list[tuple[str, Callable[[Sequence[ScenarioOutcome]], bool] | None, str]] = [
        ("all cells", None, DEFAULT_SELECTOR),
        # Holding the cap constant isolates effort from finishing: if this survives, the selector
        # is not just re-reading max_steps truncation.
        ("both episodes finished", all_finished, "more-replies"),
        ("exactly one finished", one_finished, DEFAULT_SELECTOR),
    ]
    for label, cell_filter, feature in controls:
        collected = []
        logger.info("\n  %s  (selector: %s)", label, feature)
        for corpus, matrix in loaded:
            try:
                bound = selector_bound(
                    matrix, corpus, feature, SELECTOR_KEYS[feature], cell_filter=cell_filter
                )
            except ValueError:
                continue
            collected.append(bound)
            logger.info(
                "    %-19s cells %3d | random %.3f selector %.3f oracle %.3f | %3d decisive,"
                " %3.0f%% correct | harvest %+4.0f%%",
                corpus,
                bound.cells,
                bound.random_of_n,
                bound.selector_accuracy,
                bound.oracle_of_n,
                bound.decisive_cells,
                bound.correct_fraction * 100,
                bound.harvested_fraction * 100,
            )
        correct, decisive, z = pooled_correct_z(collected)
        logger.info(
            "    POOLED %d/%d = %.1f%% correct, z vs coin flip = %+.2f",
            correct,
            decisive,
            correct / decisive * 100 if decisive else 0.0,
            z,
        )

    logger.info("\n" + "=" * 100)
    logger.info("(c) CORRELATION DECOMPOSITION  (pooled ~= between-cell => difficulty, not quality)")
    logger.info("=" * 100)
    logger.info("\n%-19s %-18s %8s %8s %8s", "corpus", "feature", "pooled", "between", "within")
    for corpus, matrix in loaded:
        for row in feature_correlations(matrix, corpus):
            logger.info(
                "%-19s %-18s %+8.3f %+8.3f %+8.3f",
                row.corpus,
                row.feature,
                row.pooled,
                row.between_cell,
                row.within_cell,
            )


if __name__ == "__main__":
    main()
