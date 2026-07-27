"""Build the routing-ablation dashboard: run records -> one self-contained HTML file.

Every number on the page is a SEED AGGREGATE. A "result" is one (matrix, variant, knobs)
group summarised across its split seeds: mean accuracy +- sd, mean cost, mean p50, n seeds.
Individual seed runs appear only inside an expander or hover.

Two data hazards this builder normalises, both of which used to make the page read as "1x
seeds" even though five seeds were run:

1. `params` mixes KNOBS with per-seed FIT OUTPUTS (`best_epoch`, `platt`, `val_bce`, the
   auto-selected config, best-single's winning `model`). Grouping on raw params therefore
   split every seed into its own group. A param key is classified as a fit output, per
   (matrix, variant), when its value is constant WITHIN each seed but differs ACROSS seeds:
   that is the signature of something the fitter produced rather than something the sweep
   set. Fit outputs are dropped from the group key and listed in the group's hover.
2. The same (group, seed) run is sometimes appended more than once (identical metrics, new
   run_id). Only the latest ts per seed survives, so n seeds and the sd are honest.

Significance is PAIRED BY SEED against the `best_single` baseline carried on the same record
(same split, same scenarios): mean delta +- sd of the per-seed deltas, plus a seed win count.
With five seeds there is no honest p-value, so the tiers are win-count and delta-vs-spread:
BEATS (delta > 0, wins >= 80% of seeds, delta > sd), WORSE (mirror image), ties (|delta| <=
sd), otherwise mixed. Groups with fewer than three seeds are badged underpowered, never
silently shown as results.

Brand-styled (white surface, ink #0a0a0a, hairline #ececec). Charts colour by FAMILY, not by
run: black diamond = best-single anchor, blue = rank, purple = irt / learned predictor,
teal = r1 retrieval (jisi/kNN), amber = r2 prox / shrinkage, red = multi-call (bo2, posthoc,
ensemble, l2d), light gray = everything else drawn underneath. A variant's knob sweep is ONE
connected curve ordered by cost, with the lam=0 / guarded point ringed.

Usage: uv run python .agents/scripts/build_dashboard.py [--out .wmo/evals/dashboard.html]
       [--all]  also include the fitter-validation matrices (other papers' model pools)
"""

from __future__ import annotations

import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from wmo.research.routing_corpus import routing_data

LOCAL_RUNS = Path(".wmo/evals/runs.jsonl")
SHARED_RUNS = routing_data() / "runs"
# The four live run files. master.jsonl is the mirror of LOCAL_RUNS; r1/r2/r3 are the
# specialist chats' files (same RunRecord shape, variant names namespaced per chat).
SHARED_FILES = ("master.jsonl", "r1.jsonl", "r2.jsonl", "r3.jsonl")
FOREIGN_POOLS = {"routerbench", "llmrouterbench-flagship"}  # other papers' model pools

MIN_SEEDS = 3  # below this a group is underpowered, badged, never headlined
MIN_TEST_SCENARIOS = 30  # below this a verdict is capped at a promising/unfavourable candidate
WIN_SHARE = 0.8  # wins needed to call a paired delta directional (4/5)
MAX_COLOURED = 12  # coloured series per chart; the rest go gray underneath

FAMILIES: dict[str, dict[str, str]] = {
    "single": {"label": "best single model (anchor)", "color": "#0a0a0a"},
    "rank": {"label": "rank / rank-tilt (cluster scoreboard)", "color": "#0070f3"},
    "irt": {"label": "irt / logistic (learned predictor)", "color": "#7928ca"},
    "knn": {"label": "r1 retrieval (jisi, kNN)", "color": "#0d9488"},
    "prox": {"label": "r2 prox / shrinkage", "color": "#b8770a"},
    "multi": {"label": "multi-call (bo2, posthoc, ensemble, l2d)", "color": "#ee0000"},
    "other": {"label": "other / controls", "color": "#b9b9b9"},
}


def family_of(variant: str) -> str:
    """Map a (possibly chat-namespaced) variant name onto a colour family."""
    v = variant.lower()
    if re.search(r"shuffl|control", v):
        return "other"
    if re.search(r"best-single|costaware-single|static", v):
        return "single"
    if re.search(r"bo2|posthoc|l2d|(^|-)ens($|-)|cascade", v):
        return "multi"
    if re.search(r"prox|shrink|(^|-)eb\d*($|-)", v):
        return "prox"
    if re.search(r"irt|logistic|(^|-)cov($|-)", v):
        return "irt"
    if "rank" in v:
        return "rank"
    if re.search(r"jisi|knn", v):
        return "knn"
    return "other"


def load_runs() -> tuple[list[dict], list[str]]:
    """Read every run record once. Local runs.jsonl wins over its master.jsonl mirror."""
    notes: list[str] = []
    sources: list[tuple[str, Path]] = []
    have_local = LOCAL_RUNS.is_file()
    if have_local:
        sources.append(("local runs.jsonl", LOCAL_RUNS))
    for name in SHARED_FILES:
        if name == "master.jsonl" and have_local:
            notes.append("skipped master.jsonl (mirror of the local runs.jsonl already read)")
            continue
        path = SHARED_RUNS / name
        if path.is_file():
            sources.append((name, path))
        else:
            notes.append(f"missing {name}")

    runs: list[dict] = []
    seen: set[str] = set()
    for label, path in sources:
        kept = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record["run_id"] in seen:  # belt and braces against a double-counted file
                continue
            seen.add(record["run_id"])
            record["src"] = label
            runs.append(record)
            kept += 1
        notes.append(f"{label}: {kept} runs")
    if SHARED_RUNS.is_dir():
        ignored = sorted(p.name for p in SHARED_RUNS.glob("*.jsonl") if p.name not in SHARED_FILES)
        if ignored:
            notes.append("not a live run file, ignored: " + ", ".join(ignored))
    return runs, notes


def resolve_matrices(runs: list[dict]) -> tuple[list[str], list[str]]:
    """Merge matrix names that are the SAME captured test set, and split apart the ones that
    are not. Rewrites each run's `matrix` in place; returns (merge notes, baseline anomalies).

    Two naming conventions collided: master/r1/r3 prefix the world-model corpora with `wm-`,
    while r2 writes the bare corpus name and puts its holdout construction in a `split` param
    (iid, ood-cluster, ood-task). So `wm-tau-bench` and `tau-bench` are partly the same matrix
    and partly not, and the bare name alone mixes two different test sets.

    One evidence-based rule settles it. For every (matrix name, split) cohort, take the modal
    best-single baseline accuracy per seed - the baseline is a property of the test set, so two
    cohorts are the same test set exactly when those vectors are equal. Cohorts that agree, and
    whose names match up to the `wm-` prefix, become one section; cohorts that disagree stay
    separate and get their split named in the section title. Nothing is merged on the say-so of
    a naming convention, and a split that was never verified equal is never silently pooled.

    Using the MODAL baseline (not the set of them) keeps one stray run from vetoing a merge;
    such strays are returned as anomalies and shown on the page instead of being swallowed.
    """
    # A (name, split) can still hide two test sets: wm-all was captured at 43 scenarios/seed
    # and later at 63, same name, different baselines. Where the test-set SIZE predicts the
    # baseline within a seed, it is a real second population, so it joins the cohort key.
    sizes: dict[tuple[str, str], dict[int, dict[int, set[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(set))
    )
    for run in runs:
        base = run.get("baselines", {}).get("best_single")
        if base:
            key = (run["matrix"], str(run["params"].get("split")))
            sizes[key][run["split_seed"]][run["test_scenarios"]].add(round(base["accuracy"], 9))

    def stable_populations(per_seed: dict[int, dict[int, set[float]]]) -> bool:
        """True when the test-set sizes look like two separate CAPTURES rather than one
        protocol with a variable holdout: the same set of sizes must recur at every seed,
        more than one of them, and each size must pin down a single baseline. An ood-cluster
        sweep whose holdout size wobbles from seed to seed fails this and stays pooled."""
        seen = [frozenset(per_n) for per_n in per_seed.values()]
        return (
            len(seen) > 1
            and len(set(seen)) == 1
            and len(seen[0]) > 1
            and all(len(accs) == 1 for per_n in per_seed.values() for accs in per_n.values())
        )

    split_by_size = {key: stable_populations(per_seed) for key, per_seed in sizes.items()}

    def cohort_key(run: dict) -> tuple:
        key = (run["matrix"], str(run["params"].get("split")))
        return (*key, run["test_scenarios"]) if split_by_size.get(key) else key

    cells: dict[tuple, dict[int, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for run in runs:
        base = run.get("baselines", {}).get("best_single")
        if base:
            cells[cohort_key(run)][run["split_seed"]][round(base["accuracy"], 9)] += 1

    modal = {
        key: {seed: counts.most_common(1)[0][0] for seed, counts in seeds.items()}
        for key, seeds in cells.items()
    }
    # What the cohort key could NOT separate: a cell where runs disagree about their own
    # baseline. The modal value wins so a stray cannot veto a merge, but say so out loud.
    anomalies: list[str] = []
    for key, per_seed in sorted(cells.items(), key=lambda kv: str(kv[0])):
        for seed, counts in sorted(per_seed.items()):
            if len(counts) > 1:
                ranked = counts.most_common()
                rest = ", ".join(f"{acc:.4f} x{n}" for acc, n in ranked[1:])
                anomalies.append(
                    f"{' · '.join(str(k) for k in key)} · seed {seed}: "
                    f"{len(counts)} different best-single baselines recorded - "
                    f"modal {ranked[0][0]:.4f} x{ranked[0][1]} used, also seen {rest}"
                )

    def bare(name: str) -> str:
        return name[3:] if name.startswith("wm-") else name

    cohorts: dict[tuple[str, tuple], list[tuple]] = defaultdict(list)
    for key, per_seed in modal.items():
        cohorts[(bare(key[0]), tuple(sorted(per_seed.items())))].append(key)

    runs_in: Counter[tuple[str, tuple]] = Counter()
    member_of: dict[tuple, tuple[str, tuple]] = {}
    for ident, members in cohorts.items():
        for member in members:
            member_of[member] = ident
    for run in runs:
        ident = member_of.get(cohort_key(run))
        if ident:
            runs_in[ident] += 1

    by_base: dict[str, list[tuple[str, tuple]]] = defaultdict(list)
    for ident in cohorts:
        by_base[ident[0]].append(ident)

    rename: dict[tuple, str] = {}
    notes: list[str] = []
    for base_name, idents in sorted(by_base.items()):
        sizes_vary = len({m[2] for i in idents for m in cohorts[i] if len(m) > 2}) > 1
        # The biggest cohort keeps the plain name; the rest say how they differ.
        for rank, ident in enumerate(sorted(idents, key=lambda i: -runs_in[i])):
            members = cohorts[ident]
            names = sorted({m[0] for m in members})
            splits = sorted({m[1] for m in members if m[1] != "None"})
            scenarios = sorted({m[2] for m in members if len(m) > 2})
            # Only drop the wm- prefix when a merge actually happened across the two names.
            label = base_name if len(names) > 1 else names[0]
            marks = []
            if rank and splits:
                marks.append("/".join(splits))
            if rank and sizes_vary and scenarios:
                marks.append(f"n={'/'.join(str(s) for s in scenarios)}/seed")
            if rank and not marks:
                marks.append(f"cohort {rank + 1}")
            if marks:
                label = f"{label} [{', '.join(marks)}]"
            for member in members:
                rename[member] = label
            if len(names) > 1:
                notes.append(
                    f"merged {' + '.join(names)} into '{label}' "
                    f"(identical per-seed best-single baselines on all {len(ident[1])} seeds; "
                    f"{runs_in[ident]} runs)"
                )
            elif rank:
                notes.append(
                    f"kept '{label}' separate from '{base_name}': its per-seed baselines differ, "
                    f"so it is a different test set ({runs_in[ident]} runs)"
                )

    for run in runs:
        key = cohort_key(run)
        if key in rename:
            run["matrix"] = rename[key]
        # The split is now part of the section identity; leave it out of the knob label.
        run["params"].pop("split", None)
    return notes, anomalies


def knob_keys(runs: list[dict]) -> dict[tuple[str, str], set[str]]:
    """Per (matrix, variant), the param keys that are KNOBS rather than per-seed fit outputs.

    A key is a fit output when its value is constant within every seed but varies across
    seeds (see module docstring). Knob sweeps vary within a seed, so they survive.
    """
    by_variant: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for run in runs:
        by_variant[(run["matrix"], run["variant"])].append(run)

    keys: dict[tuple[str, str], set[str]] = {}
    for ident, rows in by_variant.items():
        every = set().union(*[set(r["params"]) for r in rows]) if rows else set()
        fitted = set()
        for key in every:
            per_seed: dict[int, set[str]] = defaultdict(set)
            for row in rows:
                per_seed[row["split_seed"]].add(json.dumps(row["params"].get(key), sort_keys=True))
            if len(per_seed) < 2 or any(len(v) > 1 for v in per_seed.values()):
                continue
            if len({next(iter(v)) for v in per_seed.values()}) > 1:
                fitted.add(key)
        keys[ident] = every - fitted
    return keys


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def knob_label(knobs: dict) -> str:
    if not knobs:
        return "(defaults)"
    return " ".join(f"{k}={json.dumps(v)}" for k, v in sorted(knobs.items()))


def tier_of(
    delta: float,
    delta_sd: float,
    wins: int,
    losses: int,
    seeds: int,
    deltas: list[float],
    test_scenarios: float,
) -> str:
    """The honest label for a paired-by-seed delta at n<=5 seeds. No invented p-values.

    Power has two axes and a verdict needs both. Seeds bound how repeatable the delta is;
    test scenarios bound how precise each seed's measurement is. A +18pt delta that wins 5/5
    seeds on SEVEN test scenarios per seed is our weakest evidence class, not a win, so it is
    capped at a promising/unfavourable candidate label rather than earning BEATS or WORSE.
    """
    if all(d == 0.0 for d in deltas):
        # Not a tie in the statistical sense: this group scored the baseline exactly, on every
        # seed, which is what a guard that always abstains to best-single looks like.
        return "identical"
    if seeds < MIN_SEEDS:
        return "underpowered"
    need = math.ceil(WIN_SHARE * seeds)
    if abs(delta) <= delta_sd:
        return "ties"
    directional = (delta > 0 and wins >= need) or (delta < 0 and losses >= need)
    if directional and test_scenarios < MIN_TEST_SCENARIOS:
        return "promising" if delta > 0 else "unfavourable"
    if delta > 0 and wins >= need:
        return "beats"
    if delta < 0 and losses >= need:
        return "worse"
    return "mixed"


def aggregate(
    runs: list[dict], knobs_by_variant: dict[tuple[str, str], set[str]]
) -> tuple[list[dict], int]:
    """Collapse runs into one seed-aggregated group record per (matrix, variant, knobs)."""
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    knob_view: dict[tuple[str, str, str], dict] = {}
    fitted_view: dict[tuple[str, str, str], list[str]] = {}
    for run in runs:
        ident = (run["matrix"], run["variant"])
        allowed = knobs_by_variant[ident]
        knobs = {k: v for k, v in run["params"].items() if k in allowed}
        key = (run["matrix"], run["variant"], json.dumps(knobs, sort_keys=True))
        grouped[key].append(run)
        knob_view[key] = knobs
        fitted_view[key] = sorted(set(run["params"]) - allowed)

    collapsed = 0
    groups: list[dict] = []
    for key, rows in grouped.items():
        per_seed: dict[int, dict] = {}
        for run in sorted(rows, key=lambda r: r["ts"]):
            if run["split_seed"] in per_seed:
                collapsed += 1
            per_seed[run["split_seed"]] = run
        kept = [per_seed[s] for s in sorted(per_seed)]
        results = [r["result"] for r in kept]
        accs = [r["accuracy"] for r in results]
        costs = [r["cost_per_call"] for r in results]
        p50s = [r["latency_p50_s"] for r in results if r.get("latency_p50_s") is not None]
        test_n = statistics.median([float(r["test_scenarios"]) for r in kept])

        paired = [
            (r["result"]["accuracy"] - r["baselines"]["best_single"]["accuracy"], r)
            for r in kept
            if r.get("baselines", {}).get("best_single")
        ]
        deltas = [d for d, _ in paired]
        wins = sum(1 for d in deltas if d > 0)
        losses = sum(1 for d in deltas if d < 0)
        base_rows = [r["baselines"]["best_single"] for _, r in paired]
        base_lat = [b["latency_p50_s"] for b in base_rows if b.get("latency_p50_s") is not None]

        mix: dict[str, float] = defaultdict(float)
        tokens: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        model_cost: dict[str, list[float]] = defaultdict(list)
        model_p50: dict[str, list[float]] = defaultdict(list)
        for result in results:
            for model, share in result["model_mix"].items():
                mix[model] += share / len(results)
            for model, bucket in (result.get("tokens_by_model") or {}).items():
                tokens[model][0] += bucket["input"]
                tokens[model][1] += bucket["output"]
            for model, value in (result.get("per_model_cost_per_call") or {}).items():
                model_cost[model].append(value)
            for model, value in (result.get("per_model_latency_p50_s") or {}).items():
                model_p50[model].append(value)

        matrix, variant, _ = key
        groups.append(
            {
                "m": matrix,
                "v": variant,
                "fam": family_of(variant),
                "kn": knob_view[key],
                "lab": knob_label(knob_view[key]),
                "fit": fitted_view[key],
                "s": len(kept),
                "seedlist": sorted(per_seed),
                "acc": _mean(accs),
                "sd": _sd(accs),
                "cost": _mean(costs),
                "csd": _sd(costs),
                "p50": _mean(p50s) if p50s else None,
                "p50sd": _sd(p50s) if p50s else 0.0,
                "nt": _mean([float(r["scenarios"]) for r in results]),
                "ntmed": test_n,
                "src": sorted({r.get("src", "?") for r in kept}),
                "unscored": _mean([float(r["unscored"]) for r in results]),
                "cps": _mean([r.get("calls_per_scenario", 1.0) for r in results]),
                "d": _mean(deltas) if deltas else None,
                "dsd": _sd(deltas) if deltas else 0.0,
                "w": wins,
                "l": losses,
                "tier": (
                    tier_of(_mean(deltas), _sd(deltas), wins, losses, len(kept), deltas, test_n)
                    if deltas
                    else "unpaired"
                ),
                # Single-model mix matching the baseline: the guard abstained everywhere.
                "abst": len(mix) == 1
                and set(mix) == {m for b in base_rows for m in b["model_mix"]},
                "b": {
                    "acc": _mean([b["accuracy"] for b in base_rows]) if base_rows else None,
                    "cost": _mean([b["cost_per_call"] for b in base_rows]) if base_rows else None,
                    "p50": _mean(base_lat) if base_lat else None,
                    # A single-model baseline's mix has exactly one entry: the winning model.
                    "models": sorted({m for b in base_rows for m in b["model_mix"]}),
                },
                "seeds": [
                    {
                        "seed": r["split_seed"],
                        "acc": r["result"]["accuracy"],
                        "cost": r["result"]["cost_per_call"],
                        "p50": r["result"].get("latency_p50_s"),
                        "n": r["result"]["scenarios"],
                        "d": (
                            r["result"]["accuracy"] - r["baselines"]["best_single"]["accuracy"]
                            if r.get("baselines", {}).get("best_single")
                            else None
                        ),
                    }
                    for r in kept
                ],
                "mix": dict(sorted(mix.items(), key=lambda kv: -kv[1])),
                "tok": dict(tokens),
                "pmc": {m: _mean(v) for m, v in model_cost.items()},
                "pml": {m: _mean(v) for m, v in model_p50.items()},
                "notes": kept[-1].get("notes", ""),
            }
        )
    return groups, collapsed


def synthetic_anchors(runs: list[dict], groups: list[dict]) -> list[dict]:
    """Surface the best-single reference on matrices where no best-single run was recorded.

    Those matrices still carry the baseline on every record - it is what their paired deltas
    are measured against, and it is unique per (matrix, seed) - so the anchor is real data,
    just never written as its own row. Without this the chart would have no reference point.
    """
    have = {g["m"] for g in groups if g["fam"] == "single"}
    per_matrix: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        base = run.get("baselines", {}).get("best_single")
        if base and run["matrix"] not in have:
            per_matrix[run["matrix"]][run["split_seed"]].append(base)

    anchors: list[dict] = []
    for matrix, by_seed in per_matrix.items():
        seeds = sorted(by_seed)
        accs = [_mean([b["accuracy"] for b in by_seed[s]]) for s in seeds]
        costs = [_mean([b["cost_per_call"] for b in by_seed[s]]) for s in seeds]
        lats = [
            _mean(v)
            for s in seeds
            if (v := [b["latency_p50_s"] for b in by_seed[s] if b.get("latency_p50_s") is not None])
        ]
        flat = [b for s in seeds for b in by_seed[s]]
        mix: dict[str, float] = defaultdict(float)
        tokens: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        model_cost: dict[str, list[float]] = defaultdict(list)
        model_p50: dict[str, list[float]] = defaultdict(list)
        for base in flat:
            for model, share in base["model_mix"].items():
                mix[model] += share / len(flat)
            for model, bucket in (base.get("tokens_by_model") or {}).items():
                tokens[model][0] += bucket["input"]
                tokens[model][1] += bucket["output"]
            for model, value in (base.get("per_model_cost_per_call") or {}).items():
                model_cost[model].append(value)
            for model, value in (base.get("per_model_latency_p50_s") or {}).items():
                model_p50[model].append(value)
        anchors.append(
            {
                "m": matrix,
                "v": "best-single (paired baseline)",
                "fam": "single",
                "kn": {},
                "lab": "reconstructed from the baseline on every run of this matrix",
                "fit": ["model"],
                "s": len(seeds),
                "seedlist": seeds,
                "acc": _mean(accs),
                "sd": _sd(accs),
                "cost": _mean(costs),
                "csd": _sd(costs),
                "p50": _mean(lats) if lats else None,
                "p50sd": _sd(lats) if lats else 0.0,
                "nt": _mean([float(b["scenarios"]) for b in flat]),
                "ntmed": statistics.median([float(b["scenarios"]) for b in flat]),
                "src": ["reconstructed from paired baselines"],
                "unscored": _mean([float(b["unscored"]) for b in flat]),
                "cps": _mean([b.get("calls_per_scenario", 1.0) for b in flat]),
                "d": None,
                "dsd": 0.0,
                "w": 0,
                "l": 0,
                "tier": "anchor",
                "abst": False,
                "b": {"acc": None, "cost": None, "p50": None, "models": sorted(mix)},
                "seeds": [
                    {
                        "seed": s,
                        "acc": _mean([b["accuracy"] for b in by_seed[s]]),
                        "cost": _mean([b["cost_per_call"] for b in by_seed[s]]),
                        "p50": by_seed[s][0].get("latency_p50_s"),
                        "n": by_seed[s][0]["scenarios"],
                        "d": None,
                    }
                    for s in seeds
                ],
                "mix": dict(sorted(mix.items(), key=lambda kv: -kv[1])),
                "tok": dict(tokens),
                "pmc": {m: _mean(v) for m, v in model_cost.items()},
                "pml": {m: _mean(v) for m, v in model_p50.items()},
                "notes": "",
            }
        )
    return anchors


def choose_series(groups: list[dict]) -> dict[str, list[str]]:
    """Per matrix, the variants that get a family colour (at most MAX_COLOURED).

    Round-robin across families so one crowded family cannot take the whole budget; within a
    family, variants rank by tier (proven results first) then by mean paired delta. The
    best-single anchor is always drawn and is not part of the budget.
    """
    rank = {
        "beats": 0,
        "mixed": 1,
        "ties": 2,
        "identical": 3,
        "worse": 4,
        "underpowered": 5,
        "anchor": 6,
        "unpaired": 7,
    }
    chosen: dict[str, list[str]] = {}
    for matrix in {g["m"] for g in groups}:
        best: dict[str, dict[str, tuple]] = defaultdict(dict)
        for g in groups:
            if g["m"] != matrix or g["fam"] in {"single", "other"}:
                continue
            score = (rank.get(g["tier"], 9), -(g["d"] or -1.0), -g["s"])
            prior = best[g["fam"]].get(g["v"])
            if prior is None or score < prior:
                best[g["fam"]][g["v"]] = score
        queues = {
            fam: [v for v, _ in sorted(items.items(), key=lambda kv: kv[1])]
            for fam, items in best.items()
        }
        picked: list[str] = []
        while len(picked) < MAX_COLOURED and any(queues.values()):
            for fam in sorted(queues):
                if queues[fam] and len(picked) < MAX_COLOURED:
                    picked.append(queues[fam].pop(0))
        chosen[matrix] = picked
    return chosen


def summary_table(groups: list[dict]) -> str:
    """The significant-vs-within-spread census, per matrix. Printed and embedded in the page."""
    order = [
        "beats",
        "promising",
        "ties",
        "worse",
        "unfavourable",
        "mixed",
        "identical",
        "underpowered",
        "anchor",
        "unpaired",
    ]
    per_matrix: dict[str, Counter] = defaultdict(Counter)
    powered: Counter[str] = Counter()
    test_n: dict[str, float] = {}
    for g in groups:
        per_matrix[g["m"]][g["tier"]] += 1
        test_n[g["m"]] = g["ntmed"]
        if g["s"] >= MIN_SEEDS and g["ntmed"] >= MIN_TEST_SCENARIOS:
            powered[g["m"]] += 1
    head = f"{'matrix':<34}{'n/seed':>7}{'groups':>7}{'full power':>11}" + "".join(
        f"{t:>14}" for t in order
    )
    lines = [head, "-" * len(head)]
    for matrix in sorted(per_matrix, key=lambda m: -sum(per_matrix[m].values())):
        counts = per_matrix[matrix]
        lines.append(
            f"{matrix:<34}{test_n.get(matrix, 0):>7.0f}{sum(counts.values()):>7}"
            f"{powered[matrix]:>11}" + "".join(f"{counts[t]:>14}" for t in order)
        )
    totals = sum(per_matrix.values(), Counter())
    lines.append("-" * len(head))
    lines.append(
        f"{'ALL':<34}{'':>7}{sum(totals.values()):>7}{sum(powered.values()):>11}"
        + "".join(f"{totals[t]:>14}" for t in order)
    )
    lines += [
        "",
        f"beats/worse = paired delta outside its own sd with >= {WIN_SHARE:.0%} of seeds agreeing.",
        "ties = |paired delta| <= sd. mixed = outside the spread but the seeds disagree.",
        "identical = zero delta on EVERY seed (a guard that always abstained to best-single);",
        f"that needs no power, so it keeps the label under {MIN_SEEDS} seeds, while the '3+ seeds'",
        "column counts groups actually powered enough for a beats/ties/worse verdict.",
        "anchor = the best-single reference itself, which has no delta against itself.",
        "",
        f"A verdict needs BOTH axes of power: {MIN_SEEDS}+ seeds AND a median test set of",
        f"{MIN_TEST_SCENARIOS}+ scenarios/seed. A directional delta on a smaller test set is",
        "capped at promising (positive) or unfavourable (negative): a candidate for the scaled",
        "capture round, never a win. 'full power' counts groups clearing both axes.",
    ]
    return "\n".join(lines)


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Routing ablations</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--ink:#0a0a0a;--muted:#6b6b6b;--grid:#ececec;--surface:#ffffff;
--c0:#0070f3;--c1:#b8770a;--c2:#7928ca;--c3:#0d9488;--c4:#ee0000;--c5:#c2007a;
--other:#b9b9b9}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);
font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;padding:32px 40px 80px}
h1{font-size:20px;margin:0 0 4px;text-align:left}
h2{font-size:15px;margin:36px 0 10px;text-align:left}
h3{font-size:13px;margin:0 0 4px;text-align:left}
.sub{color:var(--muted);margin:0 0 20px;max-width:980px}
.filters{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0 8px}
.filters button{border:1px solid var(--grid);background:var(--surface);color:var(--ink);
padding:5px 12px;border-radius:6px;cursor:pointer;font:inherit}
.filters button.on{border-color:var(--ink);font-weight:600}
.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--muted);margin:6px 0 2px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.swatch{width:10px;height:10px;border-radius:3px;display:inline-block;flex:none}
.serieskey{color:var(--muted);font-size:11.5px;margin:4px 0 0;line-height:1.7}
.serieskey b{color:var(--ink);font-weight:600}
svg text{font:11px -apple-system,sans-serif;fill:var(--muted)}
.pop{position:fixed;pointer-events:none;background:var(--surface);border:1px solid var(--grid);border-radius:12px;box-shadow:0 10px 30px rgba(0,0,0,.10);padding:16px 18px;max-width:420px;z-index:10;opacity:0;transition:opacity .1s;font-size:12.5px;line-height:1.55}
.pop h4{margin:0 0 8px;font-size:13px}
.pop table{margin:0;width:100%}.pop td,.pop th{padding:2px 8px 2px 0;border:none;text-align:right;font-size:12px}
.pop td:first-child,.pop th:first-child{text-align:left}
.tip{position:fixed;pointer-events:none;background:var(--ink);color:#fff;padding:8px 10px;
border-radius:6px;font-size:12px;line-height:1.5;opacity:0;transition:opacity .08s;max-width:360px;z-index:9}
table{border-collapse:collapse;width:100%;margin-top:8px}
th,td{padding:6px 10px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
th{color:var(--muted);font-weight:500}
tr.seedrow td{color:var(--muted);font-size:12px;background:#fcfcfc}
tr.grouprow{cursor:pointer}
tr.grouprow:hover td{background:#fafafa}
.grid2{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:28px}
.note{border:1px solid var(--grid);border-radius:8px;padding:12px 16px;color:var(--muted);margin-top:10px}
.info{display:inline-flex;align-items:center;justify-content:center;width:14px;height:14px;border:1px solid var(--muted);border-radius:50%;color:var(--muted);font-size:10px;margin-left:5px;cursor:help;vertical-align:1px}
.badge{display:inline-block;border-radius:5px;padding:1px 7px;font-size:11px;font-weight:600;
border:1px solid currentColor;white-space:nowrap}
.tile{border:1px solid var(--grid);border-radius:10px;padding:14px 16px;cursor:help}
pre.census{border:1px solid var(--grid);border-radius:8px;padding:12px 16px;overflow-x:auto;
font:11.5px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--muted)}
</style></head><body>
<h1>Routing ablations</h1>
<p class="sub">Cost, quality and speed per router variant on OUR model pool. Every point,
tile and table row is a <b>seed aggregate</b>: one (matrix, variant, knobs) group summarised
across its split seeds as mean +- sd. Verdicts are <b>paired by seed</b> against the
best-single baseline recorded on the same split, reported as mean delta +- sd plus a seed win
count; with five seeds that framing is honest and a p-value would not be. Rebuild with --all
to also see the fitter-validation matrices (other papers' model pools).</p>
<div class="filters" id="matrixFilter"></div>
<div class="legend" id="familyLegend"></div>
<div class="legend" style="margin-top:4px"><span><i class="swatch" style="background:var(--ink)"></i>filled = significant (paired delta outside the seed spread, __WINSHARE__ of seeds agree)</span><span><i class="swatch" style="background:var(--surface);border:2px solid var(--muted)"></i>hollow = within spread / mixed seeds</span><span>ringed = lam 0 (guarded) point of a knob sweep</span></div>
<h2>Headlines: best FULL-POWER routed group vs best single model</h2>
<div id="heads" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:14px"></div>
<h2>Candidates for the scaled capture round</h2>
<p class="sub" id="candNote"></p>
<div id="cands"></div>
<h2>Cost vs quality (Pareto), per matrix</h2>
<div class="grid2" id="pareto"></div>
<h2>Model mix, charted groups only</h2>
<div class="legend" id="mixLegend"></div>
<div id="mix"></div>
<h2>Blended tokens by model, charted groups only</h2>
<div id="tokens"></div>
<h2>Verdict census (every group, by matrix)</h2>
<pre class="census" id="census">__CENSUS__</pre>
<h2>Provenance: which matrix names were merged, and what did not line up</h2>
<pre class="census" id="prov">__PROVENANCE__</pre>
<h2>All groups (click a row for its seeds)</h2>
<table id="runs"></table>
<div class="note">Reading guide: cost and p50 latency often drop <b>together</b> when the mix
shifts toward cheaper models, because cheaper models are usually also faster - verify via the
per-model p50 in each hover and the token breakdown. A single seed's accuracy delta is not a
result; the paired delta +- sd and the win count are. Groups under __MINSEEDS__ seeds carry an
underpowered badge and are excluded from headlines.</div>
<div class="tip" id="tip"></div>
<div class="pop" id="pop"></div>
<script>
const DATA = __DATA__;
const SERIES = __SERIES__;
const FAM = __FAMILIES__;
const MIN_SEEDS = __MINSEEDS__;
const MIN_TEST = __MINTEST__;
const LOADING = __LOADING__;
const MATRIX_INFO = {
 'routerbench': 'RouterBench (2024): the classic public matrix. 36k prompts x 11 models from 2023 (gpt-4, claude-2 era) with measured per-call costs. Validates the fitter against published conventions; models are dated.',
 'llmrouterbench-flagship': 'LLMRouterBench (2026) flagship track: 13 modern flagship models (gemini-2.5, gpt-5, claude-sonnet-4, deepseek, qwen3-235b...) on hard datasets (AIME, GPQA, HLE...). Deduped by task text after a caught leak; small (809 scenarios), so noise floors are wide.',
 'routerbench-ours9': 'OUR 9-model pool (gpt-5.5/5.4-mini, fable/sonnet/haiku/opus, deepseek-v4-pro, kimi-k2.6, glm-5.2) run live on 1,199 gold-certified RouterBench MCQ prompts, exact-match graded (judge-free). Real latency per call.',
 'routerbench-ours9-duptraffic': 'The ours9 matrix with duplicated traffic: the same prompts repeated to test whether a router exploits repetition rather than task structure.',
 'headroom3-all': 'Headroom probe across three corpora: how much accuracy an oracle router could reach, i.e. the ceiling any real router is chasing.',
};
function matrixInfo(m){
 if(MATRIX_INFO[m]) return MATRIX_INFO[m];
 if(m.startsWith('wm-')) return 'Closed-loop world-model scenarios: our 9-model pool rolled as the agent against the '+m.slice(3)+' base world model (judge pinned Opus 4.8, tool surface derived from the corpus traces). Small per-corpus test sides: read the cross-corpus aggregate, not one row.';
 return 'Live agent runs on the '+m+' corpus with our 9-model pool, scored by the pinned judge. Held-out split per seed.';
}
const COL_INFO = {
 'matrix':'Which outcome matrix (benchmark) the group was evaluated on. Hover the benchmark filters above for descriptions.',
 'variant':'Router variant + its KNOBS. Per-seed fitted values (best_epoch, platt, the auto-chosen config) are not knobs and are excluded from the grouping; the row hover lists them. lam = the cost knob: reward points paid per average-call-cost unit (0 = pure accuracy).',
 'seeds':'How many split seeds this group aggregates. Under '+MIN_SEEDS+' seeds it is badged underpowered.',
 'acc':'Mean accuracy across seeds +- sd across seeds, on HELD-OUT scenarios.',
 'delta':'PAIRED delta: for each seed, this group minus the best-single baseline measured on the SAME split, then averaged (+- sd of those per-seed deltas).',
 'wins':'Seeds where the paired delta was positive, out of the seeds in the group.',
 'verdict':'BEATS = delta > 0, wins >= '+Math.round(100*__WINSHARERAW__)+'% of seeds, and delta > its own sd. ties = |delta| <= sd. WORSE = the mirror of BEATS. mixed = delta outside the spread but the seeds disagree. No p-values: n is 5.',
 'cost/call':'Mean measured cost per call in USD, averaged over seeds (per-call usage x the real per-token price; never list-price guesses).',
 'vs cost':'Cost delta (%) vs the paired best-single baseline. Negative = cheaper.',
 'p50':'Median latency of the routed calls, averaged over seeds. "-" = this matrix carries no timings.',
 'n/seed':'Held-out test scenarios per seed. Small n = wide noise floor (~+-1/sqrt(n)).',
};
const VARIANT_INFO = {
 'best-single': 'No routing: always use whichever ONE model scored best overall on the fit data. The bar every router must beat. Which model that is can differ per seed, so it is a fitted value, not a knob.',
 'costaware-single': 'Like best-single but picking the single model with the best accuracy-per-dollar on the fit data instead of the best raw accuracy.',
 'rank': 'Embed each task, cluster them, keep a per-cluster scoreboard of which model wins there; a new request goes to its nearest cluster’s champion. (Avengers, arXiv 2505.19797)',
 'rank-tilt': 'Same as rank, but trust big clusters more than tiny ones - a weird one-off request should not be routed by a 3-scenario cluster’s noisy scoreboard. (ProxRouter-inspired, arXiv 2510.09852)',
 'irt': 'No clusters: learn a small report card per model (what it’s good at) and a difficulty score per question, then predict who passes - like matching students to exam questions. Needs much less data. (IrtNet, arXiv 2510.00844)',
 'logistic': 'The same predict-who-passes idea as irt with a plain logistic model on the embedding, as the honest simple-baseline control for it.',
 'jisi': 'Find the most similar past tasks, throw out false matches (neighbors where models answered very differently from the rest - looked alike, wasn’t alike), then route to whoever did best on what remains. (JiSi proxy mode, arXiv 2601.01330)',
 'knn': 'Route by nearest neighbours: score each model on the retrieved similar tasks and send the request to the leader, with a statistical guard that abstains to best-single when the lead is inside the noise.',
 'prox': 'Neighbour scoreboards shrunk toward the global average (empirical-Bayes / proximity weighting) so a thin neighbourhood cannot outvote the pool-wide evidence.',
 'static': 'A fixed hand-picked model for everything.',
 'bo2-free': 'Does not choose a model from the question at all: runs ONE model twice and keeps the better attempt - the one that finished rather than running out of steps, and did more work getting there. Costs 2x per request, so it only switches on when doubling up beat the single best model on the fit data by a clear margin; otherwise it falls back to one call. Judging which attempt is better is free (no extra model call), and uses only how the attempt ran, never its score.',
 'l2d': 'Learning to defer: a cheap model answers unless a learned deferral rule says this request needs the expensive one.',
 'ens': 'Multi-call ensemble: several models answer and a selector picks among the attempts. Costs more than one call per request by construction.',
 'shuffled': 'NEGATIVE CONTROL: the same router with its learned signal shuffled. It should score at or below best-single; if it does not, the variant was reading noise.',
};
function variantInfo(v){
 const l=v.toLowerCase();
 const keys=['best-single','costaware-single','rank-tilt','bo2-free','shuffled','logistic','static','jisi','prox','irt','rank','knn','l2d','ens'];
 for(const k of keys) if(l.includes(k)) return VARIANT_INFO[k]||'';
 return '';
}
const tip = document.getElementById('tip');
const pop = document.getElementById('pop');
function showPop(e, html){pop.innerHTML=html;pop.style.opacity=1;
 const w=Math.min(420,innerWidth-40);
 pop.style.left=Math.min(e.clientX+16,innerWidth-w-20)+'px';
 pop.style.top=Math.max(10,Math.min(e.clientY+14,innerHeight-pop.offsetHeight-20))+'px';}
function hidePop(){pop.style.opacity=0;}
function showTip(e, html){tip.innerHTML=html;tip.style.opacity=1;
 tip.style.left=Math.min(e.clientX+14,innerWidth-380)+'px';tip.style.top=(e.clientY+12)+'px';}
function hideTip(){tip.style.opacity=0;}
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;');}
const fmt$ = v=>'$'+v.toFixed(5), fmtP=v=>(100*v).toFixed(1)+'%';
const fmtPt = v=>(v>=0?'+':'')+(100*v).toFixed(1)+'pt';
// One global colour per model (fixed by overall token volume), shared by every section.
const MODEL_COLORS={};
{const vol={};
 DATA.forEach(g=>Object.entries(g.tok||{}).forEach(([m,b])=>vol[m]=(vol[m]||0)+b[0]+b[1]));
 const GR=['#5c5c5c','#7d7d7d','#9a9a9a','#b3b3b3','#c9c9c9','#dedede','#6e6e6e','#8b8b8b'];
 Object.keys(vol).sort((a,b)=>vol[b]-vol[a]).forEach((m,i)=>MODEL_COLORS[m]=i<5?`var(--c${i})`:GR[(i-5)%GR.length]);}
const TIER={
 beats:{text:'BEATS baseline',color:'var(--c3)',fill:true},
 worse:{text:'WORSE than baseline',color:'var(--c4)',fill:true},
 promising:{text:'promising (small test set)',color:'var(--c1)',fill:false},
 unfavourable:{text:'unfavourable (small test set)',color:'var(--c1)',fill:false},
 ties:{text:'ties baseline (within spread)',color:'var(--muted)',fill:false},
 identical:{text:'identical to baseline (never routed away)',color:'var(--muted)',fill:false},
 mixed:{text:'mixed seeds (unclear)',color:'var(--c1)',fill:false},
 underpowered:{text:'underpowered',color:'var(--c1)',fill:false},
 anchor:{text:'baseline (reference)',color:'var(--ink)',fill:true},
 unpaired:{text:'no paired baseline',color:'var(--muted)',fill:false}};
// The small-test tiers carry their n in the text: the number IS the caveat.
function tierText(g){const t=TIER[g.tier]||TIER.unpaired;
 return (g.tier==='promising'||g.tier==='unfavourable')
  ?t.text.replace(')',`, n=${Math.round(g.ntmed)})`):t.text;}
function fullPower(g){return g.s>=MIN_SEEDS && g.ntmed>=MIN_TEST;}
function famColor(f){return (FAM[f]||FAM.other).color;}
// 'other' is a 40%-opacity gray on charts; as label ink it would be unreadable.
function labelColor(f){return f==='other'?'var(--ink)':famColor(f);}
function isColoured(g){return g.fam==='single' || (SERIES[g.m]||[]).includes(g.v);}
function deltaText(g){
 if(g.d==null) return 'no paired baseline';
 return `${fmtPt(g.d)} <span style="color:var(--muted)">+- ${(100*g.dsd).toFixed(1)}</span> · ${g.w}/${g.s} seeds win`;
}
function powerBadge(g){
 let b='';
 if(g.s<MIN_SEEDS) b+=` <span class="badge" style="color:var(--c1)">n=${g.s} seed${g.s>1?'s':''} · underpowered</span>`;
 if(g.ntmed<MIN_TEST) b+=` <span class="badge" style="color:var(--c1)">${Math.round(g.ntmed)} scenarios/seed · small test set</span>`;
 return b;
}
function tooltip(g){
 const t=TIER[g.tier]||TIER.unpaired;
 let h=`<b>${esc(g.v)}</b><br><span style="opacity:.8">${esc(g.lab)}</span>`+
  `<br>${esc(g.m)} · ${g.s} seed${g.s>1?'s':''} · ${Math.round(g.nt)} scenarios/seed`+
  `<br>accuracy <b>${fmtP(g.acc)} +- ${(100*g.sd).toFixed(1)}</b> · cost ${fmt$(g.cost)}`+
  (g.csd>0?` +- ${fmt$(g.csd)}`:'')+
  (g.p50?`<br>p50 ${g.p50.toFixed(2)}s`:'')+
  `<br>vs best-single (paired): ${deltaText(g)}`+
  `<br><b>${tierText(g)}</b>`;
 if(g.s<MIN_SEEDS) h+=`<br><span style="color:#ffd479">underpowered: fewer than ${MIN_SEEDS} seeds</span>`;
 if(g.cps>1.01) h+=`<br>${g.cps.toFixed(2)} calls per scenario (cost is per scenario)`;
 if(g.fit&&g.fit.length) h+=`<br><span style="opacity:.75">fitted per seed, not a knob: ${esc(g.fit.join(', '))}</span>`;
 const mix=Object.entries(g.mix).slice(0,4).map(([m,s])=>{const p=g.pml[m];
   return `${esc(m)} ${fmtP(s)}${p?` (p50 ${p.toFixed(2)}s)`:''}`}).join('<br>');
 return h+'<br><br><u>mean mix</u><br>'+mix;
}
function paretoChart(matrix){
 const groups=DATA.filter(g=>g.m===matrix);
 const W=560,H=330,L=56,R=96,T=14,B=42;
 const lo=v=>Math.max(v,1e-9);
 const costs=groups.map(g=>g.cost).filter(v=>v>0);
 const accLo=Math.min(...groups.map(g=>g.acc-g.sd)), accHi=Math.max(...groups.map(g=>g.acc+g.sd));
 const x0=lo(Math.min(...costs))/1.6, x1=Math.max(...costs)*1.6;
 const y0=Math.max(0,accLo-0.02), y1=Math.min(1,accHi+0.02);
 const X=v=>L+(Math.log(lo(v))-Math.log(x0))/(Math.log(x1)-Math.log(x0))*(W-L-R);
 const Y=v=>T+(y1-v)/(y1-y0||1)*(H-T-B);
 let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="cost vs accuracy by family, ${esc(matrix)}">`;
 for(let i=0;i<=4;i++){const v=y0+(y1-y0)*i/4;
  s+=`<line x1="${L}" x2="${W-R}" y1="${Y(v).toFixed(1)}" y2="${Y(v).toFixed(1)}" stroke="var(--grid)"/>`+
     `<text x="${L-6}" y="${(Y(v)+4).toFixed(1)}" text-anchor="end">${fmtP(v)}</text>`;}
 for(let k=0;k<8;k++){const v=x0*Math.pow(4,k); if(v>=x1) break;
  s+=`<text x="${X(v).toFixed(1)}" y="${H-B+16}" text-anchor="middle">${fmt$(v)}</text>`;}
 s+=`<text x="${((L+W-R)/2).toFixed(0)}" y="${H-6}" text-anchor="middle">cost per call (log)</text>`;
 const draw=(g,coloured)=>{
  const cx=X(g.cost), cy=Y(g.acc), col=coloured?famColor(g.fam):'var(--other)';
  const op=coloured?1:0.4, t=TIER[g.tier]||TIER.unpaired;
  let out='';
  if(g.sd>0) out+=`<line x1="${cx.toFixed(1)}" x2="${cx.toFixed(1)}" y1="${Y(Math.min(y1,g.acc+g.sd)).toFixed(1)}" y2="${Y(Math.max(y0,g.acc-g.sd)).toFixed(1)}" stroke="${col}" stroke-width="1.25" opacity="${op*0.75}"/>`;
  if(g.csd>0.02*g.cost) out+=`<line y1="${cy.toFixed(1)}" y2="${cy.toFixed(1)}" x1="${X(Math.max(x0,g.cost-g.csd)).toFixed(1)}" x2="${X(Math.min(x1,g.cost+g.csd)).toFixed(1)}" stroke="${col}" stroke-width="1.25" opacity="${op*0.75}"/>`;
  const anchor=g.fam==='single'&&/best-single/.test(g.v);
  const guarded=coloured&&(g.kn.lam===0||g.kn.lam===undefined)&&!anchor;
  if(guarded) out+=`<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="8.5" fill="none" stroke="${col}" stroke-width="1" opacity="${op*0.5}"/>`;
  const fill=t.fill?col:'var(--surface)';
  const mark=anchor
   ? `<path d="M${cx.toFixed(1)},${(cy-6.5).toFixed(1)}L${(cx+6.5).toFixed(1)},${cy.toFixed(1)}L${cx.toFixed(1)},${(cy+6.5).toFixed(1)}L${(cx-6.5).toFixed(1)},${cy.toFixed(1)}Z" fill="var(--ink)" stroke="var(--surface)" stroke-width="1.5"/>`
   : `<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${coloured?5.5:4}" fill="${fill}" stroke="${col}" stroke-width="${t.fill?1.5:2}" opacity="${op}"/>`;
  return out+`<g data-i="${DATA.indexOf(g)}" style="cursor:pointer">${mark}</g>`;
 };
 // Gray series first (underneath), then coloured knob curves, then the anchor on top.
 const coloured=SERIES[matrix]||[];
 const grays=groups.filter(g=>!isColoured(g));
 s+=grays.map(g=>draw(g,false)).join('');
 const key=[];
 for(const variant of coloured){
  const pts=groups.filter(g=>g.v===variant).sort((a,b)=>a.cost-b.cost);
  if(!pts.length) continue;
  const col=famColor(pts[0].fam);
  if(pts.length>1){
   const d=pts.map((g,i)=>`${i?'L':'M'}${X(g.cost).toFixed(1)},${Y(g.acc).toFixed(1)}`).join('');
   s+=`<path d="${d}" fill="none" stroke="${col}" stroke-width="1.75" opacity="0.55"/>`;}
  s+=pts.map(g=>draw(g,true)).join('');
  const last=pts[pts.length-1];
  key.push({v:variant,fam:pts[0].fam,col:col,x:X(last.cost),y:Y(last.acc)});
 }
 groups.filter(g=>g.fam==='single').forEach(g=>{s+=draw(g,true);});
 // End-of-series labels, nudged apart so they stay readable.
 key.sort((a,b)=>a.y-b.y).forEach((k,i,arr)=>{
  if(i&&k.y-arr[i-1].y<11) k.y=arr[i-1].y+11;
  s+=`<text x="${Math.min(k.x+9,W-4).toFixed(1)}" y="${(k.y+3.5).toFixed(1)}" fill="${k.col}" style="font-size:9.5px">${esc(k.v)}</text>`;});
 s+=`</svg>`;
 const fams=[...new Set(coloured.map(v=>groups.find(g=>g.v===v).fam))];
 const legend=fams.map(f=>`<span><i class="swatch" style="background:${famColor(f)}"></i><b>${esc(FAM[f].label)}</b>: `+
   coloured.filter(v=>groups.find(g=>g.v===v).fam===f).map(v=>esc(v)).join(', ')+'</span>').join(' &nbsp; ');
 const div=document.createElement('div');
 const powered=groups.filter(g=>g.s>=MIN_SEEDS).length;
 div.innerHTML=`<h3>${esc(matrix)}<span class="info" data-t="${esc(matrixInfo(matrix))}">i</span></h3>`+
  `<div style="color:var(--muted);font-size:11.5px;margin:0 0 6px">${groups.length} groups · ${powered} with ${MIN_SEEDS}+ seeds · ${coloured.length} coloured series, ${grays.length} groups in gray</div>`+
  s+`<div class="serieskey">${legend}</div>`;
 const inf=div.querySelector('.info');
 inf.addEventListener('mousemove',e=>showTip(e,inf.dataset.t));
 inf.addEventListener('mouseleave',hideTip);
 div.querySelectorAll('g[data-i]').forEach(c=>{
  c.addEventListener('mousemove',e=>showTip(e,tooltip(DATA[+c.dataset.i])));
  c.addEventListener('mouseleave',hideTip);});
 return div;
}
function charted(){ // the anchor + coloured series of every active matrix
 return DATA.filter(g=>active.has(g.m)&&isColoured(g));
}
function stacked(containerId, per, unit){
 const cont=document.getElementById(containerId); cont.innerHTML='';
 const rows=charted();
 const share={};
 rows.forEach(g=>Object.entries(per(g)).forEach(([m,v])=>share[m]=(share[m]||0)+v));
 const models=Object.keys(share).sort((a,b)=>share[b]-share[a]);
 if(containerId==='mix'){document.getElementById('mixLegend').innerHTML=
  models.map(m=>`<span><i class="swatch" style="background:${MODEL_COLORS[m]||'var(--other)'}"></i>${esc(m)}</span>`).join('');}
 for(const g of rows){
  const entries=Object.entries(per(g)); if(!entries.length) continue;
  const total=entries.reduce((a,[,v])=>a+v,0); if(!total) continue;
  const W=520,H=22; let x=0, s=`<svg viewBox="0 0 ${W} ${H}" style="max-width:${W}px">`;
  entries.sort((a,b)=>b[1]-a[1]).forEach(([m,v])=>{const w=v/total*(W-2);
   s+=`<rect x="${x.toFixed(1)}" y="2" width="${Math.max(w-2,1).toFixed(1)}" height="16" rx="3" fill="${MODEL_COLORS[m]||'var(--other)'}"
    data-t="${esc(m)}: ${unit==='%'?fmtP(v/total):Math.round(v).toLocaleString()}"/>`;x+=w;});
  s+='</svg>';
  const row=document.createElement('div');
  row.style.cssText='display:flex;gap:12px;align-items:center;margin:3px 0';
  row.innerHTML=`<span style="width:360px;color:var(--muted);font-size:11.5px;text-align:right;overflow:hidden;text-overflow:ellipsis">${esc(g.m)} · <b style="color:${labelColor(g.fam)}">${esc(g.v)}</b> ${esc(g.lab)}</span>`+s;
  row.querySelectorAll('rect').forEach(el=>{
   el.addEventListener('mousemove',e=>showTip(e,el.dataset.t));
   el.addEventListener('mouseleave',hideTip);});
  cont.appendChild(row);
 }
}
function mixDetail(g){
 const W=380;
 const calls={}; Object.entries(g.mix).forEach(([m,s])=>calls[m]=s);
 const totCost={}; Object.entries(g.pmc||{}).forEach(([m,c])=>totCost[m]=c*(calls[m]||0));
 const costSum=Object.values(totCost).reduce((a,v)=>a+v,0)||1;
 const tokSum=Object.values(g.tok||{}).reduce((a,b)=>a+b[0]+b[1],0)||1;
 const models=Object.keys(g.mix).sort((a,b)=>(totCost[b]||0)-(totCost[a]||0));
 let seg='', x=0;
 models.forEach(m=>{const w=(totCost[m]||0)/costSum*W;
  if(w>0.5) seg+=`<rect x="${x.toFixed(1)}" y="2" width="${Math.max(w-2,1).toFixed(1)}" height="12" rx="3" fill="${MODEL_COLORS[m]||'var(--other)'}"/>`; x+=w;});
 const rows=models.map(m=>{const bk=(g.tok||{})[m]||[0,0];
  return `<tr><td><i class="swatch" style="background:${MODEL_COLORS[m]||'var(--other)'}"></i> ${esc(m)}</td>`+
   `<td>${fmtP(g.mix[m])}</td><td>${fmtP((bk[0]+bk[1])/tokSum)}</td>`+
   `<td>${fmt$(totCost[m]||0)}</td><td>${fmtP((totCost[m]||0)/costSum)}</td></tr>`}).join('');
 return `<div style="font-weight:600;margin:10px 0 4px">Cost by model (mean mix)</div>`+
  `<svg viewBox="0 0 ${W} 16" style="width:100%;display:block">${seg}</svg>`+
  `<table style="margin-top:6px"><tr><th>model</th><th>calls</th><th>tok %</th><th>cost</th><th>cost %</th></tr>${rows}</table>`;
}
function seedTable(g){
 return `<table><tr><th>seed</th><th>acc</th><th>paired delta</th><th>cost</th><th>n</th></tr>`+
  g.seeds.map(s=>`<tr><td>seed ${s.seed}</td><td>${fmtP(s.acc)}</td>`+
   `<td>${s.d==null?'-':fmtPt(s.d)}</td><td>${fmt$(s.cost)}</td><td>${s.n}</td></tr>`).join('')+`</table>`;
}
function detailPop(g){
 const t=TIER[g.tier]||TIER.unpaired;
 return `<h4>${esc(g.v)} <span style="color:var(--muted);font-weight:400">${esc(g.lab)}</span></h4>`+
  `<div style="color:var(--muted);margin-bottom:6px">${esc(g.m)} · ${g.s} seeds (${g.seedlist.join(', ')}) · ${Math.round(g.nt)} scenarios/seed</div>`+
  `<table><tr><th></th><th>acc</th><th>cost/call</th><th>p50</th></tr>`+
  `<tr><td>this group</td><td><b>${fmtP(g.acc)} +- ${(100*g.sd).toFixed(1)}</b></td><td><b>${fmt$(g.cost)}</b></td><td><b>${g.p50?g.p50.toFixed(2)+'s':'-'}</b></td></tr>`+
  `<tr><td>best-single${g.b.models.length?' · '+esc(g.b.models.join('/')):''}</td><td>${g.b.acc!=null?fmtP(g.b.acc):'-'}</td><td>${g.b.cost!=null?fmt$(g.b.cost):'-'}</td><td>${g.b.p50?g.b.p50.toFixed(2)+'s':'-'}</td></tr></table>`+
  `<div style="margin:8px 0 10px;padding:6px 10px;border:1px solid var(--grid);border-radius:6px">`+
  `<b style="color:${t.color}">${tierText(g)}</b> · paired ${deltaText(g)}`+
  (g.s<MIN_SEEDS?`<br><b style="color:var(--c1)">n=${g.s} seeds - underpowered, do not read as a result</b>`:'')+
  (g.nt<60?`<br><b style="color:var(--c1)">${Math.round(g.nt)} scenarios/seed - wide noise floor (~+-${(100/Math.sqrt(Math.max(g.nt,1))).toFixed(1)}pt)</b>`:'')+
  (g.fit&&g.fit.length?`<br><span style="color:var(--muted)">fitted per seed (excluded from the group key): ${esc(g.fit.join(', '))}</span>`:'')+
  (g.cps>1.01?`<br><span style="color:var(--muted)">${g.cps.toFixed(2)} calls/scenario - cost is per scenario</span>`:'')+
  `</div>`+
  seedTable(g)+mixDetail(g)+
  (variantInfo(g.v)?`<div style="margin-top:10px;color:var(--muted)">${esc(variantInfo(g.v))}</div>`:'')+
  (g.notes?`<div style="margin-top:8px;color:var(--muted)">${esc(g.notes)}</div>`:'');
}
function headlines(){
 const cont=document.getElementById('heads'); cont.innerHTML='';
 const rank={beats:0,promising:1,mixed:2,ties:3,identical:4,unfavourable:5,worse:6};
 const skipped=[];
 for(const m of matrices.filter(x=>active.has(x))){
  // Full power only: a tile is a claim, and a claim needs seeds AND scenarios behind it.
  const rows=DATA.filter(g=>g.m===m&&g.fam!=='single'&&g.d!=null&&fullPower(g)).slice()
   .sort((a,b)=>(rank[a.tier]??9)-(rank[b.tier]??9)||b.d-a.d);
  if(!rows.length){skipped.push(m); continue;}
  const g=rows[0], t=TIER[g.tier]||TIER.unpaired;
  const xCost=(g.b.cost&&g.cost)?g.b.cost/g.cost:null;
  const xLat=(g.p50&&g.b.p50)?g.b.p50/g.p50:null;
  const good=v=>v>=0?'var(--c3)':'var(--c4)';
  const div=document.createElement('div');
  div.className='tile';
  div.innerHTML=`<div style="color:var(--muted);font-size:12px">${esc(m)} · ${g.s} seeds · ${Math.round(g.nt)} scenarios/seed</div>
   <div style="font-size:13px;font-weight:600;margin:4px 0 0;color:${labelColor(g.fam)}">${esc(g.v)}<span class="info" data-t="${esc(variantInfo(g.v))}">i</span></div>
   <div style="font-size:24px;font-weight:650;margin:2px 0 2px">${fmtP(g.acc)}<span style="font-size:12px;color:var(--muted)"> +- ${(100*g.sd).toFixed(1)}</span></div>
   <div style="font-size:12.5px">vs best-single: <b style="color:${good(g.d)}">${fmtPt(g.d)}</b> <span style="color:var(--muted)">+- ${(100*g.dsd).toFixed(1)}</span> · <b>${g.w}/${g.s} seeds win</b></div>
   <div style="margin:5px 0 4px"><span class="badge" style="color:${t.color}">${tierText(g)}</span>${powerBadge(g)}</div>
   <div style="color:var(--muted)">`+
   (xCost?`<b style="color:${good(xCost-1)}">${xCost>=1?xCost.toFixed(1)+'x cheaper':(1/xCost).toFixed(1)+'x pricier'}</b>`:'')+
   (xLat?` · <b style="color:${good(xLat-1)}">${xLat>=1?xLat.toFixed(1)+'x faster':(1/xLat).toFixed(1)+'x slower'}</b>`:'')+`</div>`;
  const inf=div.querySelector('.info');
  inf.addEventListener('mousemove',e=>{e.stopPropagation();showTip(e,inf.dataset.t);});
  inf.addEventListener('mouseleave',hideTip);
  div.addEventListener('mousemove',e=>showPop(e,detailPop(g)));
  div.addEventListener('mouseleave',hidePop);
  cont.appendChild(div);
 }
 candidates(skipped);
}
function candidates(skipped){
 // Everything the power gate kept out of the headlines: the queue for the scaled capture round.
 const note=document.getElementById('candNote'), cont=document.getElementById('cands');
 cont.innerHTML='';
 const rank={promising:0,mixed:1,ties:2,identical:3,unfavourable:4};
 const rows=[];
 for(const m of matrices.filter(x=>active.has(x))){
  const pool=DATA.filter(g=>g.m===m&&g.fam!=='single'&&g.d!=null&&!fullPower(g)&&g.s>=MIN_SEEDS)
   .sort((a,b)=>(rank[a.tier]??9)-(rank[b.tier]??9)||b.d-a.d);
  if(pool.length) rows.push(pool[0]);
 }
 note.innerHTML=`${skipped.length} of ${matrices.filter(x=>active.has(x)).length} matrices have no `+
  `full-power group (median test set under ${MIN_TEST} scenarios per seed), so they get no headline `+
  `tile. Their strongest directional group is listed here as a candidate to re-measure in the `+
  `scaled capture round - these are hypotheses, not results.`;
 if(!rows.length){cont.innerHTML='<div class="note">No candidates.</div>'; return;}
 rows.sort((a,b)=>(rank[a.tier]??9)-(rank[b.tier]??9)||b.d-a.d);
 const t=document.createElement('table');
 t.innerHTML='<tr><th>matrix</th><th>best directional group</th><th>seeds</th><th>n/seed</th>'+
  '<th>paired delta</th><th>wins</th><th>status</th></tr>'+
  rows.map(g=>{const ti=TIER[g.tier]||TIER.unpaired;
   return `<tr class="grouprow" data-i="${DATA.indexOf(g)}"><td>${esc(g.m)}</td>`+
    `<td><b style="color:${labelColor(g.fam)}">${esc(g.v)}</b> <span style="color:var(--muted)">${esc(g.lab)}</span></td>`+
    `<td>${g.s}</td><td><b style="color:var(--c1)">${Math.round(g.ntmed)}</b></td>`+
    `<td>${fmtPt(g.d)} <span style="color:var(--muted)">+- ${(100*g.dsd).toFixed(1)}</span></td>`+
    `<td>${g.w}/${g.s}</td><td style="color:${ti.color}">${tierText(g)}</td></tr>`}).join('');
 t.querySelectorAll('tr.grouprow').forEach(tr=>{
  const g=DATA[+tr.dataset.i];
  tr.addEventListener('mousemove',e=>showPop(e,detailPop(g)));
  tr.addEventListener('mouseleave',hidePop);});
 cont.appendChild(t);
}
function table(){
 const t=document.getElementById('runs');
 const rank={beats:0,promising:1,mixed:2,ties:3,identical:4,unfavourable:5,worse:6,
  underpowered:7,anchor:8,unpaired:9};
 const rows=DATA.filter(g=>active.has(g.m)).slice().sort((a,b)=>
  matrices.indexOf(a.m)-matrices.indexOf(b.m)||a.fam.localeCompare(b.fam)||
  a.v.localeCompare(b.v)||(rank[a.tier]??9)-(rank[b.tier]??9)||b.acc-a.acc);
 const th=(label,key)=>`<th>${label}<span class="info" data-t="${esc(COL_INFO[key]||'')}">i</span></th>`;
 t.innerHTML='<tr>'+th('matrix','matrix')+th('variant · knobs','variant')+th('seeds','seeds')+
  th('acc +- sd','acc')+th('paired delta','delta')+th('wins','wins')+th('verdict','verdict')+
  th('cost/call','cost/call')+th('vs cost','vs cost')+th('p50','p50')+th('n/seed','n/seed')+'</tr>'+
  rows.map(g=>{const ti=TIER[g.tier]||TIER.unpaired;
   const costPct=(g.b.cost&&g.cost)?(100*(g.cost/g.b.cost-1)).toFixed(0)+'%':'-';
   return `<tr class="grouprow" data-i="${DATA.indexOf(g)}">`+
    `<td>${esc(g.m)}</td>`+
    `<td><b style="color:${labelColor(g.fam)}">${esc(g.v)}</b> <span style="color:var(--muted)">${esc(g.lab)}</span></td>`+
    `<td>${g.s}${g.s<MIN_SEEDS?' <span class="badge" style="color:var(--c1)">underpowered</span>':''}</td>`+
    `<td>${fmtP(g.acc)} <span style="color:var(--muted)">+- ${(100*g.sd).toFixed(1)}</span></td>`+
    `<td>${g.d==null?'-':fmtPt(g.d)+' <span style="color:var(--muted)">+- '+(100*g.dsd).toFixed(1)+'</span>'}</td>`+
    `<td>${g.d==null?'-':g.w+'/'+g.s}</td>`+
    `<td style="color:${ti.color};font-weight:${ti.fill?600:400}">${tierText(g)}</td>`+
    `<td>${fmt$(g.cost)}</td><td>${costPct}</td>`+
    `<td>${g.p50?g.p50.toFixed(2)+'s':'-'}</td><td>${Math.round(g.nt)}</td></tr>`}).join('');
 t.querySelectorAll('th .info').forEach(el=>{
  el.addEventListener('mousemove',e=>showTip(e,el.dataset.t));
  el.addEventListener('mouseleave',hideTip);});
 t.querySelectorAll('tr.grouprow').forEach(tr=>{
  const g=DATA[+tr.dataset.i];
  tr.addEventListener('mousemove',e=>showPop(e,detailPop(g)));
  tr.addEventListener('mouseleave',hidePop);
  tr.addEventListener('click',()=>{
   if(tr.nextSibling&&tr.nextSibling.classList&&tr.nextSibling.classList.contains('seedrow')){
    while(tr.nextSibling&&tr.nextSibling.classList&&tr.nextSibling.classList.contains('seedrow')) tr.nextSibling.remove();
    return;}
   g.seeds.slice().reverse().forEach(s=>{
    const r=document.createElement('tr'); r.className='seedrow';
    r.innerHTML=`<td></td><td>seed ${s.seed}</td><td>1</td><td>${fmtP(s.acc)}</td>`+
     `<td>${s.d==null?'-':fmtPt(s.d)}</td><td></td><td>single seed, not a result</td>`+
     `<td>${fmt$(s.cost)}</td><td></td><td>${s.p50?s.p50.toFixed(2)+'s':'-'}</td><td>${s.n}</td>`;
    tr.after(r);});
  });});
}
const matrices=[...new Set(DATA.map(g=>g.m))];
let active=new Set(matrices);
function render(){
 headlines();
 const p=document.getElementById('pareto'); p.innerHTML='';
 matrices.filter(m=>active.has(m)).forEach(m=>p.appendChild(paretoChart(m)));
 stacked('mix', g=>g.mix, '%');
 stacked('tokens', g=>Object.fromEntries(Object.entries(g.tok).map(([m,b])=>[m,b[0]+b[1]])), 'tok');
 table();
}
const mf=document.getElementById('matrixFilter');
matrices.forEach(m=>{const b=document.createElement('button');b.className='on';
 b.innerHTML=esc(m)+`<span class="info" data-t="${esc(matrixInfo(m))}">i</span>`;
 b.onclick=()=>{b.classList.toggle('on');b.classList.contains('on')?active.add(m):active.delete(m);render();};
 const i=b.querySelector('.info');
 i.addEventListener('mousemove',e=>{e.stopPropagation();showTip(e,i.dataset.t);});
 i.addEventListener('mouseleave',hideTip);
 i.addEventListener('click',e=>e.stopPropagation());
 mf.appendChild(b);});
document.getElementById('familyLegend').innerHTML=Object.entries(FAM)
 .map(([f,d])=>`<span data-f="${f}" style="cursor:help"><i class="swatch" style="background:${d.color}${f==='other'?';opacity:.4':''}"></i>${esc(d.label)}</span>`).join('')+
 `<span class="info" data-t="${esc('Loaded: '+LOADING.join(' | '))}">i</span>`;
document.querySelectorAll('#familyLegend .info').forEach(el=>{
 el.addEventListener('mousemove',e=>showTip(e,el.dataset.t));
 el.addEventListener('mouseleave',hideTip);});
// All guarded: when a router is not clearly confident it beats best-single, it does not try.
render();
</script></body></html>
"""


def main() -> None:
    out = (
        Path(sys.argv[sys.argv.index("--out") + 1])
        if "--out" in sys.argv
        else Path(".wmo/evals/dashboard.html")
    )
    runs, notes = load_runs()
    if not runs:
        raise SystemExit(f"no run records found (looked in {LOCAL_RUNS} and {SHARED_RUNS})")
    if "--all" not in sys.argv:
        runs = [r for r in runs if r["matrix"] not in FOREIGN_POOLS]
        notes.append(f"default view: dropped {sorted(FOREIGN_POOLS)} (pass --all to keep them)")

    merge_notes, anomalies = resolve_matrices(runs)
    notes += merge_notes
    groups, collapsed = aggregate(runs, knob_keys(runs))
    anchors = synthetic_anchors(runs, groups)
    groups += anchors
    notes.append(f"collapsed {collapsed} duplicate (group, seed) runs")
    if anchors:
        matrices = sorted({a["m"] for a in anchors})
        notes.append(
            "reconstructed the best-single anchor from paired baselines on "
            + ", ".join(matrices)
            + " (no best-single run of their own)"
        )
    series = choose_series(groups)
    over = {m: v for m, v in series.items() if len(v) > MAX_COLOURED}
    assert not over, f"coloured-series budget blown: {over}"

    # Matrix order: our live pool first, then the world-model twins, biggest first.
    def matrix_rank(matrix: str) -> tuple:
        return (matrix.startswith("wm-"), -sum(1 for g in groups if g["m"] == matrix), matrix)

    order = {m: i for i, m in enumerate(sorted({g["m"] for g in groups}, key=matrix_rank))}
    groups.sort(key=lambda g: (order[g["m"]], g["fam"], g["v"], g["cost"]))

    census = summary_table(groups)
    provenance = "\n".join(
        ["Matrix-name resolution (see resolve_matrices in the builder):"]
        + [f"  {n}" for n in merge_notes]
        + [
            "",
            f"Per-run baseline anomalies: {len(anomalies)}"
            + (
                ". A run whose recorded best-single differs from every other run of the same "
                "(matrix, split, seed) means its baseline was fit differently; the merge uses "
                "the modal baseline, so one stray cannot veto it, but the stray is listed here."
                if anomalies
                else " - every run agrees with its cohort's baseline."
            ),
        ]
        + [f"  {a}" for a in anomalies]
    )
    html = (
        TEMPLATE.replace("__DATA__", json.dumps(groups))
        .replace("__SERIES__", json.dumps(series))
        .replace("__FAMILIES__", json.dumps(FAMILIES))
        .replace("__PROVENANCE__", provenance)
        .replace("__MINTEST__", str(MIN_TEST_SCENARIOS))
        .replace("__MINSEEDS__", str(MIN_SEEDS))
        .replace("__WINSHARERAW__", repr(WIN_SHARE))
        .replace("__WINSHARE__", f"{WIN_SHARE:.0%}")
        .replace("__LOADING__", json.dumps(notes))
        .replace("__CENSUS__", census)
    )
    # Syntax gate: a single bad quote blanks the whole page silently; check before shipping.
    if shutil.which("node"):
        script = re.search(r"<script>(.*)</script>", html, re.S)
        assert script is not None
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
            handle.write(script.group(1))
        subprocess.run(["node", "--check", handle.name], check=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    tiers: Counter[str] = Counter(g["tier"] for g in groups)
    sys.stderr.write("\n".join(f"  {n}" for n in notes) + "\n\n")
    sys.stderr.write(census + "\n\n")
    sys.stderr.write(
        f"dashboard: {len(runs)} runs -> {len(groups)} seed-aggregated groups "
        f"({tiers['beats']} beats, {tiers['ties']} ties, {tiers['worse']} worse, "
        f"{tiers['mixed']} mixed, {tiers['underpowered']} underpowered) -> {out}\n"
    )


if __name__ == "__main__":
    main()
