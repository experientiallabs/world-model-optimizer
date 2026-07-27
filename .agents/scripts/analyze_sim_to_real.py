"""Sim-to-real: does the world-model tau-bench matrix rank our pool the way real tau2 does?

Two matrices, two very different measurements of the same nine candidates:

- REAL  `.wmo/evals/tau-bench-real/rows.jsonl` — Sierra's tau2-bench, tau2's own deterministic
  reward (DB state + action/communicate checks). Binary per episode.
- WM    `<routing-data>/matrices/tau-bench_matrix.json` — our world model, LLM judge, continuous.

The task sets only partly overlap (10 real scenarios share a `reason_for_call` with 9 wm ones), so
the headline comparison is rank agreement over MODEL MEANS, with the paired overlap reported as a
secondary check rather than the main number.

Sampling unit is the TASK, not the episode: a model is scored twice on each task, so SE over the
per-task means (rather than over all episodes) is the one that does not pretend the two trials of
one task are independent draws.

`--glm-clean` reruns the correlations with glm-5.2's wm mean recomputed to net out its inline
tool-call format failures. Two estimators, because the obvious one is biased:

- `clean_only` simply drops the 17 broken episodes. It overstates glm badly: the format failure is
  concentrated on the HARD scenarios (the other eight models average 0.23-0.39 on the scenarios
  where glm inlined both episodes, against 0.59 where it never did), so dropping them also drops
  the tasks glm would have found hardest.
- `clean_if_available` keeps all 25 scenarios and, for the 13 with one good and one broken episode,
  scores the scenario on its clean episode only. This is the number the report uses.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path
from typing import Any

from scipy import stats
from wmo.research.routing_corpus import routing_data

REAL_ROWS = Path("/Users/silen/Desktop/Projects/wmh-optimizer-core/.wmh/evals/tau-bench-real/rows.jsonl")
WM_MATRIX = routing_data() / "matrices/tau-bench_matrix.json"

# A tool call the model wrote into its prose instead of emitting as a structured call, e.g.
# `get_user_details({"user_id": ...})`. The harness never executes these, so the episode dies.
INLINE_CALL = re.compile(
    r"\b(get|list|search|update|cancel|book|modify|return|exchange|transfer|calculate|find|send|think)_?\w*\(\s*[{\"]"
)


def load_real() -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in REAL_ROWS.read_text(encoding="utf-8").splitlines() if line]
    return [row for row in rows if row["reward"] is not None]


def load_wm() -> list[dict[str, Any]]:
    return json.loads(WM_MATRIX.read_text(encoding="utf-8"))["outcomes"]


def has_inline_call(outcome: dict[str, Any]) -> bool:
    return any(INLINE_CALL.search(reply or "") for reply in outcome.get("replies") or [])


def task_clustered_stats(values_by_task: dict[str, list[float]]) -> tuple[float, float, int]:
    """Mean of per-task means, plus the SE of that mean across tasks."""
    per_task = [st.mean(v) for v in values_by_task.values()]
    mean = st.mean(per_task)
    se = st.stdev(per_task) / len(per_task) ** 0.5 if len(per_task) > 1 else float("nan")
    return mean, se, len(per_task)


def group(rows: list[dict[str, Any]], model: str, key: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for row in rows:
        if row["model"] == model:
            out.setdefault(row[key], []).append(float(row["reward"]))
    return out


def rank_agreement(real: dict[str, float], wm: dict[str, float]) -> dict[str, Any]:
    models = sorted(real)
    r = [real[m] for m in models]
    w = [wm[m] for m in models]
    spearman = stats.spearmanr(r, w)
    kendall = stats.kendalltau(r, w)
    pearson = stats.pearsonr(r, w)
    top3_real = {m for m in sorted(real, key=real.get, reverse=True)[:3]}
    top3_wm = {m for m in sorted(wm, key=wm.get, reverse=True)[:3]}
    return {
        "spearman": (spearman.statistic, spearman.pvalue),
        "kendall": (kendall.statistic, kendall.pvalue),
        "pearson": (pearson.statistic, pearson.pvalue),
        "top3_real": sorted(top3_real),
        "top3_wm": sorted(top3_wm),
        "top3_overlap": len(top3_real & top3_wm),
        "best_real": max(real, key=real.get),
        "best_wm": max(wm, key=wm.get),
    }


def paired_scenarios(
    real: list[dict[str, Any]], wm: list[dict[str, Any]]
) -> dict[str, dict[str, list[float]]]:
    """Per-model rewards restricted to scenarios both matrices contain.

    The two runs sample tau2 differently and their scenario ids do not correspond (wm hashes the
    task, real uses "<domain>:<task_id>"), so tasks are matched on a normalized `reason_for_call`.
    One real pair (airline:15/16) shares its text with a single wm scenario and is dropped as
    ambiguous rather than double-counted.
    """
    normalize = lambda text: re.sub(r"\W+", "", text or "").lower()  # noqa: E731

    def reason(blob: str) -> str:
        try:
            return normalize(json.loads(blob).get("reason_for_call"))
        except (json.JSONDecodeError, AttributeError):
            return ""

    wm_key = {x["scenario_id"]: reason(x["task"]) for x in wm}
    real_key = {r["scenario_id"]: reason(r["task"]) for r in real}
    shared = set(wm_key.values()) & set(real_key.values()) - {""}
    ambiguous = {k for k in shared if sum(v == k for v in real_key.values()) > 1}
    shared -= ambiguous

    out: dict[str, dict[str, list[float]]] = {}
    for model in sorted({r["model"] for r in real}):
        rvals, wvals = [], []
        for key in sorted(shared):
            rv = [
                float(r["reward"])
                for r in real
                if r["model"] == model and real_key[r["scenario_id"]] == key
            ]
            wv = [
                float(x["reward"])
                for x in wm
                if x["model"] == model and wm_key[x["scenario_id"]] == key
            ]
            if rv and wv:
                rvals.append(st.mean(rv))
                wvals.append(st.mean(wv))
        out[model] = {"real": rvals, "wm": wvals}
    return out if all(v["real"] for v in out.values()) else {}


def fable_gap(scores: dict[str, float]) -> tuple[float, float]:
    """(fable - mean of the rest, fable - best of the rest)."""
    rest = [v for m, v in scores.items() if m != "fable-5"]
    return scores["fable-5"] - st.mean(rest), scores["fable-5"] - max(rest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glm-clean", action="store_true", help="also report the format-corrected arm")
    args = parser.parse_args()

    real, wm = load_real(), load_wm()
    models = sorted({row["model"] for row in real})

    print("== REAL (tau2 reward) vs WM (LLM judge), per model")
    print(f"{'model':16} {'real':>16} {'trials':>7} {'wm':>16} {'inline%':>8}")
    real_mean: dict[str, float] = {}
    wm_mean: dict[str, float] = {}
    wm_clean: dict[str, float] = {}
    table: list[dict[str, Any]] = []
    for model in models:
        rmean, rse, ntask = task_clustered_stats(group(real, model, "scenario_id"))
        neps = sum(1 for row in real if row["model"] == model)
        wmodel = [x for x in wm if x["model"] == model]
        wmean, wse, _ = task_clustered_stats(group(wmodel, model, "scenario_id"))
        clean = [x for x in wmodel if not has_inline_call(x)]
        inline_pct = 100 * (len(wmodel) - len(clean)) / len(wmodel)
        # clean_if_available: keep every scenario, but score it on its unbroken episodes when it
        # has any. Dropping broken episodes outright would also drop the hard scenarios, which is
        # where the format failure concentrates.
        by_scenario: dict[str, list[dict[str, Any]]] = {}
        for x in wmodel:
            by_scenario.setdefault(x["scenario_id"], []).append(x)
        cmean = st.mean(
            st.mean(e["reward"] for e in ([c for c in eps if not has_inline_call(c)] or eps))
            for eps in by_scenario.values()
        )
        real_mean[model], wm_mean[model], wm_clean[model] = rmean, wmean, cmean
        table.append(
            {
                "model": model,
                "real": rmean,
                "real_se": rse,
                "episodes": neps,
                "tasks": ntask,
                "wm": wmean,
                "wm_se": wse,
                "wm_clean": cmean,
                "inline_pct": inline_pct,
            }
        )
        print(
            f"{model:16} {rmean:.3f} +/- {rse:.3f} {neps:7d} {wmean:.3f} +/- {wse:.3f} {inline_pct:7.0f}%"
        )

    for label, wmscores in (("headline", wm_mean),) + (
        (("glm-format-corrected", {**wm_mean, "glm-5.2": wm_clean["glm-5.2"]}),) if args.glm_clean else ()
    ):
        agree = rank_agreement(real_mean, wmscores)
        print(f"\n== rank agreement ({label})")
        for key in ("spearman", "kendall", "pearson"):
            stat, p = agree[key]
            print(f"  {key:9} {stat:+.3f}  (p={p:.3f})")
        print(f"  top-3 real: {agree['top3_real']}")
        print(f"  top-3 wm:   {agree['top3_wm']}")
        print(f"  overlap:    {agree['top3_overlap']}/3   best real={agree['best_real']} best wm={agree['best_wm']}")
        gm, gb = fable_gap(real_mean)
        print(f"  fable gap REAL: vs mean-of-rest {gm:+.3f}, vs best-of-rest {gb:+.3f}")
        gm, gb = fable_gap(wmscores)
        print(f"  fable gap WM:   vs mean-of-rest {gm:+.3f}, vs best-of-rest {gb:+.3f}")

    print("\n== paired overlap (scenarios present in BOTH matrices)")
    paired = paired_scenarios(real, wm)
    if paired:
        preal = {m: st.mean(v["real"]) for m, v in paired.items()}
        pwm = {m: st.mean(v["wm"]) for m, v in paired.items()}
        nsc = len(next(iter(paired.values()))["real"])
        print(f"  {nsc} shared scenarios per model")
        for model in models:
            print(f"  {model:16} real {preal[model]:.3f}   wm {pwm[model]:.3f}")
        agree = rank_agreement(preal, pwm)
        print(f"  spearman {agree['spearman'][0]:+.3f} (p={agree['spearman'][1]:.3f})")

    print("\n== per-trial real means (episode noise)")
    for model in models:
        parts = []
        for ep in (0, 1):
            v = [r["reward"] for r in real if r["model"] == model and r["episode"] == ep]
            if v:
                parts.append(f"t{ep}={st.mean(v):.3f} (n={len(v)})")
        print(f"  {model:16} {'  '.join(parts)}")

    print(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
