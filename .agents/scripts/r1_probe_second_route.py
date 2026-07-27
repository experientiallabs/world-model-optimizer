"""Q3 probe: does the proxy_s2s second route drop the RIGHT neighbors?

For every ours9 test item (seed 0): compute the jisi neighbor set, the refine scores, and the
50% cut. Then judge the cut offline (test rewards used for EVALUATION only, never selection):

- dataset purity: are dropped neighbors less often from the test item's own dataset?
- profile fidelity: which subset's profile better predicts the test item's actual per-model
  rewards (Spearman-ish agreement and the regret of the subset's argmax pick)?

Writes a JSONL of per-item evidence plus a summary to stdout logging; a few concrete examples
are printed for the findings file (rule 5: read actual outputs).
"""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path

import numpy as np

from wmo.optimize.outcomes import OutcomeMatrix
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

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("r1.probe")

DATA = routing_data()
OUT = DATA / "findings" / "r1_debug" / "second_route_probe_seed0.jsonl"


def main() -> None:
    matrix = OutcomeMatrix.load(DATA / "matrices" / "routerbench-ours9_matrix.json")
    ctx = MatrixContext(matrix, "routerbench-ours9")
    fit_ids, test_ids = split_scenario_ids(matrix, train_fraction=0.7, seed=0)
    best_name, _, _ = best_single_model(matrix, fit_ids=fit_ids, eval_ids=test_ids)
    fit_matrix = np.stack([ctx.task_vecs[sid] for sid in fit_ids])

    def profile(rows_idx, weights):  # noqa: ANN001, ANN202
        scores = {}
        for model in ctx.model_names:
            num = den = 0.0
            for j, weight in zip(rows_idx, weights, strict=True):
                cell = ctx.rewards_cell.get((fit_ids[int(j)], model))
                if cell:
                    num += float(weight) * (sum(cell) / len(cell))
                    den += float(weight)
            if den:
                scores[model] = num / den
        return scores

    rows = []
    for sid in test_ids:
        sims = fit_matrix @ ctx.task_vecs[sid]
        k = min(50, len(fit_ids))
        kth = np.sort(sims)[-k]
        neighbor_rows = np.where(sims > 0.95 * kth)[0]
        if len(neighbor_rows) < 4:
            continue
        first = profile(neighbor_rows, sims[neighbor_rows])
        needles = sorted(first, key=lambda m: -first[m])[:3]
        refine = []
        for j in neighbor_rows:
            sims_resp = []
            for model in needles:
                mine = ctx.reply_vecs.get((fit_ids[int(j)], model))
                if mine is None:
                    continue
                others = [
                    ctx.reply_vecs[(fit_ids[int(o)], model)]
                    for o in neighbor_rows
                    if o != j and (fit_ids[int(o)], model) in ctx.reply_vecs
                ]
                if not others:
                    continue
                sims_resp.append(float(mine @ np.mean(others[:5], axis=0)))
            resp = float(np.mean(sims_resp)) if sims_resp else 0.0
            refine.append(0.5 * float(sims[int(j)]) + 0.5 * resp)
        refine_arr = np.asarray(refine)
        keep_n = max(1, int(len(neighbor_rows) * 0.5))
        order = np.argsort(refine_arr)
        kept, dropped = order[-keep_n:], order[:-keep_n]

        prefix = sid.split(":", 1)[0]

        def same_rate(idx, rows=neighbor_rows, pre=prefix):  # noqa: ANN001, ANN202
            ids = [fit_ids[int(rows[i])] for i in idx]
            return sum(1 for n in ids if n.split(":", 1)[0] == pre) / len(ids)

        truth = {
            m: sum(cell) / len(cell)
            for m in ctx.model_names
            if (cell := ctx.rewards_cell.get((sid, m)))
        }

        def regret(prof, tru=truth):  # noqa: ANN001, ANN202
            if not prof or not tru:
                return None
            pick = max(prof, key=lambda name: (prof[name], -ctx.model_names.index(name)))
            return max(tru.values()) - tru.get(pick, 0.0)

        kept_prof = profile(neighbor_rows[kept], refine_arr[kept])
        drop_prof = profile(neighbor_rows[dropped], refine_arr[dropped]) if len(dropped) else {}
        rows.append(
            {
                "sid": sid,
                "n": len(neighbor_rows),
                "same_prefix_kept": same_rate(kept),
                "same_prefix_dropped": same_rate(dropped) if len(dropped) else None,
                "regret_kept": regret(kept_prof),
                "regret_dropped": regret(drop_prof) if drop_prof else None,
                "regret_all": regret(first),
                "corr_refine_sim": float(np.corrcoef(refine_arr, sims[neighbor_rows])[0, 1]),
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    kept_r = [r["same_prefix_kept"] for r in rows]
    drop_r = [r["same_prefix_dropped"] for r in rows if r["same_prefix_dropped"] is not None]
    logger.info("items: %d, best_single=%s", len(rows), best_name)
    logger.info(
        "same-dataset rate: kept %.3f vs dropped %.3f",
        float(np.mean(kept_r)),
        float(np.mean(drop_r)),
    )
    for key in ("regret_kept", "regret_dropped", "regret_all"):
        vals = [r[key] for r in rows if r[key] is not None]
        logger.info("%s: mean %.4f (n=%d)", key, float(np.mean(vals)), len(vals))
    corr = [r["corr_refine_sim"] for r in rows if not np.isnan(r["corr_refine_sim"])]
    logger.info("corr(refine, query-sim): mean %.3f", float(np.mean(corr)))
    logger.info("evidence -> %s", OUT)


if __name__ == "__main__":
    main()
