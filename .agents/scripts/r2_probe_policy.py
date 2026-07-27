"""Round 12: optimal probing for the routability pre-check (pre-registered in findings/r2.md).

When trace-only signals are ambiguous, which (model, task-cluster) cells should be measured
with real API calls to decide GO/NO-GO on routing with the fewest dollars? This simulator
replays the question on our finished matrices: probing = revealing stored episodes, so the
whole policy is measurable for $0 against ground-truth verdicts.

Verdict rule (provisional, see pre-registration): H = oracle - best-single, S = share of
scenarios where some model strictly beats best-single; GO iff H >= 0.10 and S >= 0.25.

Policies:
- optimal: Beta posteriors per (model, cluster), kNN prior-transfer from the pooled wm bank,
  probe = argmax over cells of expected verdict-entropy reduction PER DOLLAR, stop when
  P(GO) exits [0.10, 0.90].
- random: uniform random unrevealed cell, same stop rule.
- grid: round-robin models x scenarios, same stop rule.

Usage: uv run python .agents/scripts/r2_probe_policy.py [corpus ...] [--reps=50]
Writes findings/r2_probe_curves.json and logs the dollars-to-verdict table.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
from wmo.research.routing_corpus import routing_data

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("r2probe")

DATA = routing_data()
RIVALS = 5  # rivals probed per champion failure, prior-ordered, stop at success
PRIOR_PSEUDO = (
    2.0  # back to 2 after amendment 2: the closed form killed the curse; data must dominate
)
H_BIN_MIN = 0.12  # amendment 2: binarized functional, H_bin == S_bin identically
MAX_PROBES = 120
MIN_PROBES = 12  # burn-in before the stop rule may fire (amendment)


def _driver():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location(
        "r2drv", Path(__file__).parent / "run_routing_r2.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def truth_verdict(matrix, anchor: str | None = None) -> tuple[bool, float, float]:  # noqa: ANN001
    """Ground-truth (GO?, H_bin, H_bin) from the full matrix (binarized functional)."""
    cells: dict[tuple[str, str], list[bool]] = {}
    for o in matrix.outcomes:
        if o.reward is not None:
            cells.setdefault((o.scenario_id, o.model), []).append(o.reward >= 0.5)
    succ = {key: float(np.mean(v)) >= 0.5 for key, v in cells.items()}
    models = [e.name for e in matrix.pool]
    sids = sorted({sid for sid, _m in succ})
    rates = {m: float(np.mean([succ.get((sid, m), False) for sid in sids])) for m in models}
    best = anchor or max(rates, key=rates.get)
    h = float(
        np.mean(
            [
                (not succ.get((sid, best), False))
                and any(succ.get((sid, m), False) for m in models if m != best)
                for sid in sids
            ]
        )
    )
    return h >= H_BIN_MIN, h, h


class PairedSim:
    """One corpus prepared for PAIRED probing (amendment 4).

    Probe unit = one episode of one model on one SCENARIO (not cluster); the policy pairs
    rivals with the champion on the same scenario because the verdict statistic is joint.
    """

    def __init__(self, name: str, matrix, prior_rates: dict[str, float]) -> None:  # noqa: ANN001
        self.name = name
        self.models = [e.name for e in matrix.pool]
        self.prior_rates = prior_rates
        self.cells: dict[tuple[str, str], list[tuple[int, float]]] = {}
        for o in matrix.outcomes:
            if o.reward is not None:
                self.cells.setdefault((o.scenario_id, o.model), []).append(
                    (int(o.reward >= 0.5), o.cost_usd)
                )
        self.sids = sorted({sid for sid, _m in self.cells})

    def probe(self, sid: str, model: str, used: dict) -> tuple[int, float] | None:
        key = (sid, model)
        index = used.get(key, 0)
        pool = self.cells.get(key, [])
        if index >= len(pool):
            return None
        used[key] = index + 1
        return pool[index]


def _jeffreys(successes: int, n: int, z: float = 1.28) -> tuple[float, float]:
    """Jeffreys-ish interval: Beta(0.5 + s, 0.5 + n - s) quantiles via normal approx."""
    a, b = 0.5 + successes, 0.5 + n - successes
    mean = a / (a + b)
    sd = float(np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1))))
    return mean - z * sd, mean + z * sd


def run_policy(sim: PairedSim, policy: str, seed: int) -> tuple[bool | None, float, list]:
    """(verdict | None, dollars, curve) for one replayed probing run."""
    rng = np.random.default_rng(seed)
    used: dict = {}
    spent = 0.0
    curve = []
    order = list(sim.sids)
    rng.shuffle(order)
    ranked = sorted(sim.models, key=lambda m: -sim.prior_rates.get(m, 0.0))

    if policy == "grid":
        # Fixed grid: every model on the first scenarios, round-robin; verdict from the
        # same X statistic once >= MIN_PROBES scenarios have full rows.
        xs = []
        for sid in order + order:  # second pass, episode-2
            outcomes = {}
            for m in sim.models:
                result = sim.probe(sid, m, used)
                if result is None:
                    continue
                outcomes[m] = result[0]
                spent += result[1]
            if not outcomes:
                continue
            best = max(outcomes, key=lambda m: sim.prior_rates.get(m, 0.0))
            best = ranked[0] if ranked[0] in outcomes else best
            xs.append(
                int(
                    outcomes.get(best, 1) == 0
                    and any(v == 1 for m2, v in outcomes.items() if m2 != best)
                )
            )
            lo, hi = _jeffreys(sum(xs), len(xs))
            curve.append((round(spent, 4), round(float(np.mean(xs)), 3)))
            if len(xs) >= MIN_PROBES and lo >= H_BIN_MIN:
                return True, spent, curve
            if len(xs) >= MIN_PROBES and hi <= H_BIN_MIN:
                return False, spent, curve
        return None, spent, curve

    # Amendment 5: the anchor is GIVEN (deploy-default = prior top-1; in production the
    # serving contract's pinned fallback). No champion estimation.
    anchor = ranked[0]
    rivals_ranked = [m for m in ranked if m != anchor]

    xs = []
    for sid in order:  # confirmation probes consume episode-2 where needed
        anchor_result = sim.probe(sid, anchor, used)
        if anchor_result is None:
            continue
        spent += anchor_result[1]
        if anchor_result[0] == 0:
            # Confirmation probe (granularity fix): a single failed episode on a noisy
            # corpus is not a failed CELL; the verdict functional is majority-of-episodes.
            confirm = sim.probe(sid, anchor, used)
            if confirm is not None:
                spent += confirm[1]
                if confirm[0] == 1:
                    anchor_result = (1, 0.0)
        if anchor_result[0] == 1:
            xs.append(0)  # anchor succeeded: no rival probe needed (short-circuit)
        else:
            rival_order = (
                rivals_ranked
                if policy == "optimal"
                else [rivals_ranked[i] for i in rng.permutation(len(rivals_ranked))]
            )
            x = 0
            for rival in rival_order[:RIVALS]:
                result = sim.probe(sid, rival, used)
                if result is None:
                    continue
                spent += result[1]
                if result[0] == 1:
                    x = 1
                    break
            xs.append(x)
        lo, hi = _jeffreys(sum(xs), len(xs))
        curve.append((round(spent, 4), round(float(np.mean(xs)), 3)))
        if len(xs) >= MIN_PROBES and lo >= H_BIN_MIN:
            return True, spent, curve
        if len(xs) >= MIN_PROBES and hi <= H_BIN_MIN:
            return False, spent, curve
    return None, spent, curve


def bank_prior_rates(drv, name: str, models: list[str]) -> dict[str, float]:  # noqa: ANN001
    """Per-model prior success rates from the wm bank, excluding the target corpus."""
    matrices = drv._matrices()
    sums: dict[str, list[float]] = {m: [] for m in models}
    for other, matrix in matrices.items():
        if other in (name, "wm-all", "routerbench-ours9") or "-s80" in other or "-real" in other:
            continue
        cells: dict[tuple[str, str], list[float]] = {}
        for o in matrix.outcomes:
            if o.reward is not None:
                cells.setdefault((o.scenario_id, o.model), []).append(o.reward)
        for (_sid, m), vals in cells.items():
            if m in sums:
                sums[m].append(float(np.mean([r >= 0.5 for r in vals])))
    return {m: float(np.mean(v)) if v else 0.5 for m, v in sums.items()}


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    reps = 50
    for flag in sys.argv[1:]:
        if flag.startswith("--reps="):
            reps = int(flag.split("=")[1])
    drv = _driver()
    matrices = drv._matrices()
    wanted = args or [
        "tau-bench",
        "financebench",
        "continual-learning",
        "bird-sql",
        "terminal-tasks",
        "crmarena",
        "dabstep",
        "swe-bench",
        "financebench-s80",
        "tau-bench-s80",
        "tau-bench-real",
    ]
    results: dict = {}
    for name in wanted:
        matrix = matrices[name]
        priors = bank_prior_rates(drv, name, [e.name for e in matrix.pool])
        anchor = max(priors, key=priors.get)  # the GIVEN deploy-default (amendment 5)
        truth, h, _s = truth_verdict(matrix, anchor=anchor)
        sim = PairedSim(name, matrix, priors)
        row: dict = {"truth": truth, "H_bin": round(h, 3), "policies": {}}
        for policy in ("optimal", "random", "grid"):
            verdicts, dollars = [], []
            for rep in range(reps):
                verdict, spent, _curve = run_policy(sim, policy, seed=rep)
                verdicts.append(verdict)
                dollars.append(spent)
            correct = float(np.mean([v == truth for v in verdicts]))
            undecided = float(np.mean([v is None for v in verdicts]))
            row["policies"][policy] = {
                "correct": round(correct, 3),
                "undecided": round(undecided, 3),
                "dollars_mean": round(float(np.mean(dollars)), 4),
                "dollars_p90": round(float(np.percentile(dollars, 90)), 4),
            }
            logger.info(
                "%-18s truth=%-5s %-7s correct=%.2f undecided=%.2f $mean=%.3f $p90=%.3f",
                name,
                "GO" if truth else "NO-GO",
                policy,
                correct,
                undecided,
                float(np.mean(dollars)),
                float(np.percentile(dollars, 90)),
            )
        results[name] = row
    out = DATA / "findings" / "r2_probe_curves.json"
    out.write_text(json.dumps(results, indent=2))
    logger.info("-> %s", out)


if __name__ == "__main__":
    main()
