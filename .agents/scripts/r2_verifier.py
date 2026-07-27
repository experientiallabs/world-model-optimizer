"""Round 11: the PAID serve-time verifier cascade (pre-registered in findings/r2.md).

A small LLM reads (task, cheap reply) and rates 0-10 whether the attempt succeeded; the
cascade escalates below a fit-calibrated threshold. FrugalGPT's actual architecture
(arXiv 2305.05176). The verifier sees ONLY what serving has at request time - task text
and the cheap transcript; never rewards or judge critiques.

Cost discipline: every verifier call is metered at the pool entry's price and cached
(atomic JSON, one process); the script HARD-STOPS at $35 cumulative spend. The verifier's
own per-episode cost is charged INSIDE the cascade's cost side, both in fit-side threshold
selection and in the replayed test rows.

Subcommands:
  score  <matrix> <cheap> <verifier> [--prompt=v0|v1] [--seed=N]   score all cheap replies
  eval   <matrix> <cheap> <strong> <verifier> [--prompt=v0|v1] [--knn]   calibrate + replay
  control <matrix> <cheap> <verifier>    shuffled-reply (mismatched task) AUC control
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.providers.base import Message
from wmo.providers.pool import PoolEntry
from wmo.providers.registry import get_provider
from wmo.research.reply_verifier import episode_key
from wmo.research.routing_runs import (
    Finish,
    RunRecord,
    append_run,
    evaluate_call_sequences,
    evaluate_choices,
)
from wmo.research.routing_corpus import routing_data

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("r2verifier")

DATA = routing_data()
RUNS = DATA / "runs" / "r2.jsonl"
CACHE = DATA / "cache" / "verifier-replies-r2.json"
SPEND_CAP = 35.0
MAX_REPLY_CHARS = 20_000
CAP_FACTOR = 0.85

V0_SYSTEM = (
    "You are a strict verifier of AI assistant work. You will see a TASK and an ATTEMPT "
    "(the assistant's full transcript). Judge whether the attempt FULLY accomplished the "
    "task. Respond with a single integer from 0 to 10 and nothing else: 0 = certainly "
    "failed, 10 = certainly fully accomplished."
)


def _load_cache() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    return {"meta": {"spent_usd": 0.0}, "calls": {}}


def _save_cache(cache: dict) -> None:
    tmp = CACHE.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(cache))
    tmp.replace(CACHE)


def _verifier_entry(matrix: OutcomeMatrix, name: str) -> PoolEntry:
    for entry in matrix.pool:
        if entry.name == name:
            return entry
    raise ValueError(f"verifier '{name}' not in the matrix pool")


def _provider(entry: PoolEntry):  # noqa: ANN202
    api_key = os.environ.get(entry.api_key_env) if entry.api_key_env else None
    return get_provider(entry.provider_config(), api_key=api_key)


def _prompt_key(verifier: str, version: str, task: str, reply: str, exemplars: str) -> str:
    blob = "|".join([verifier, version, exemplars, task, reply])
    return hashlib.sha256(blob.encode()).hexdigest()


def _reply_text(outcome: ScenarioOutcome) -> str:
    return "\n\n".join(outcome.replies)[:MAX_REPLY_CHARS]


def _exemplars(matrix: OutcomeMatrix, fit_ids: list[str], cheap: str) -> str:
    """4 extreme fit cells (2 success, 2 failure) rendered as few-shot examples (v1)."""
    cells: dict[str, list[ScenarioOutcome]] = {}
    tasks: dict[str, str] = {}
    for o in matrix.outcomes:
        tasks.setdefault(o.scenario_id, o.task)
        if o.scenario_id in set(fit_ids) and o.model == cheap and o.reward is not None:
            cells.setdefault(o.scenario_id, []).append(o)
    means = {sid: float(np.mean([o.reward for o in eps])) for sid, eps in cells.items()}
    ordered = sorted(means, key=means.get)
    picks = ordered[:2] + ordered[-2:]
    parts = []
    for sid in picks:
        outcome = cells[sid][0]
        rating = 10 if means[sid] >= 0.5 else 0
        parts.append(
            f"EXAMPLE TASK:\n{tasks[sid][:2000]}\n\nEXAMPLE ATTEMPT:\n"
            f"{_reply_text(outcome)[:3000]}\n\nRATING: {rating}"
        )
    return "\n\n---\n\n".join(parts)


def score_episodes(
    matrix: OutcomeMatrix,
    cheap: str,
    verifier: str,
    *,
    version: str = "v0",
    exemplars: str = "",
) -> tuple[dict, float]:
    """Verifier score per cheap episode key. Returns (key -> {score, cost}, run spend)."""
    entry = _verifier_entry(matrix, verifier)
    provider = _provider(entry)
    cache = _load_cache()
    tasks: dict[str, str] = {}
    for o in matrix.outcomes:
        tasks.setdefault(o.scenario_id, o.task)
    episodes = [
        o for o in matrix.outcomes if o.model == cheap and o.reward is not None and o.replies
    ]
    out: dict = {}
    spent_run = 0.0
    dirty = 0
    for o in episodes:
        key = _prompt_key(verifier, version, tasks[o.scenario_id], _reply_text(o), exemplars)
        if key in cache["calls"]:
            out[str(episode_key(o))] = cache["calls"][key]
            continue
        if cache["meta"]["spent_usd"] >= SPEND_CAP:
            raise RuntimeError(
                f"HARD STOP: cumulative verifier spend "
                f"${cache['meta']['spent_usd']:.2f} >= ${SPEND_CAP}"
            )
        body = ""
        if exemplars:
            body += exemplars + "\n\n---\n\nNow rate this one.\n\n"
        body += (
            f"TASK:\n{tasks[o.scenario_id][:6000]}\n\nATTEMPT:\n{_reply_text(o)}\n\n"
            "Rating (single integer 0-10):"
        )
        completion = provider.complete(
            # max_tokens must leave room for reasoning tokens on reasoning-capable
            # models; 8 starved gpt-5.4-mini into empty completions (64% None rate).
            V0_SYSTEM, [Message(role="user", content=body)], temperature=0.0, max_tokens=768
        )
        cost = entry.cost_usd(completion.usage)
        match = re.search(r"\b(10|[0-9])\b", completion.text)
        score = int(match.group(1)) if match else None
        record = {"score": score, "cost": cost, "text": completion.text[:40]}
        cache["calls"][key] = record
        cache["meta"]["spent_usd"] += cost
        spent_run += cost
        out[str(episode_key(o))] = record
        dirty += 1
        if dirty % 25 == 0:
            _save_cache(cache)
            logger.info(
                "scored %d/%d (run $%.3f, lifetime $%.3f)",
                len(out),
                len(episodes),
                spent_run,
                cache["meta"]["spent_usd"],
            )
    _save_cache(cache)
    logger.info(
        "%s/%s/%s: %d episodes scored, run spend $%.3f (lifetime $%.3f)",
        matrix_name(matrix),
        cheap,
        f"{verifier}-{version}",
        len(out),
        spent_run,
        cache["meta"]["spent_usd"],
    )
    return out, spent_run


_NAMES: dict[int, str] = {}


def matrix_name(matrix: OutcomeMatrix) -> str:
    return _NAMES.get(id(matrix), "matrix")


def _load_matrix(name: str) -> OutcomeMatrix:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "r2drv", Path(__file__).parent / "run_routing_r2.py"
    )
    drv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drv)
    matrix = drv._matrices()[name]
    _NAMES[id(matrix)] = name
    return matrix, drv


def evaluate(
    name: str,
    cheap: str,
    strong: str,
    verifier: str,
    *,
    version: str,
    use_knn: bool,
    seeds: list[int],
) -> None:
    matrix, drv = _load_matrix(name)
    import importlib.util as _ilu

    cspec = _ilu.spec_from_file_location("r2casc", Path(__file__).parent / "r2_cascade.py")
    casc = _ilu.module_from_spec(cspec)
    cspec.loader.exec_module(casc)

    task_vecs = casc._task_vectors(name, matrix) if use_knn else None
    ts = datetime.now(tz=UTC).isoformat()
    for seed in seeds:
        fit_ids, test_ids = drv.split_scenario_ids(matrix, train_fraction=0.7, seed=seed)
        best_name, _a, _c = drv.best_single_model(matrix, fit_ids=fit_ids, eval_ids=test_ids)
        best_eval = evaluate_choices(matrix, test_ids, lambda _sid, b=best_name: b)
        exemplars = _exemplars(matrix, fit_ids, cheap) if version == "v1" else ""
        scores, _spend = score_episodes(
            matrix, cheap, verifier, version=version, exemplars=exemplars
        )

        cells = casc._cells(matrix, set(fit_ids))
        rewards, costs = casc._fit_stats(cells)
        cap = CAP_FACTOR * float(
            np.mean([costs[(sid, best_name)] for sid in fit_ids if (sid, best_name) in costs])
        )
        est_fit = est_test = {}
        if use_knn:
            strong_means = {
                sid: rewards[(sid, strong)] for sid in fit_ids if (sid, strong) in rewards
            }
            est_fit = casc._strong_estimates(fit_ids, fit_ids, task_vecs, strong_means, loo=True)
            est_test = casc._strong_estimates(fit_ids, test_ids, task_vecs, strong_means, loo=False)

        def stat(
            sid: str, cells: dict = cells, scores: dict = scores, cheap: str = cheap
        ) -> tuple[float | None, float]:
            eps = cells.get((sid, cheap), [])
            values, vcost = [], 0.0
            for o in eps:
                rec = scores.get(str(episode_key(o)))
                if rec and rec["score"] is not None:
                    values.append(rec["score"])
                    vcost += rec["cost"]
            return (float(np.mean(values)) if values else None, vcost / max(len(eps), 1))

        rng = np.random.default_rng(0)
        resamples = [
            [fit_ids[i] for i in rng.integers(0, len(fit_ids), size=len(fit_ids))]
            for _ in range(200)
        ]
        b_grid = [None]
        if use_knn:
            s_known = [est_fit[sid] for sid in fit_ids if sid in est_fit]
            b_grid = sorted({float(q) for q in np.quantile(s_known, np.arange(0.2, 1.0, 0.2))})
        best_cfg = None
        for t in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            for b in b_grid:
                acc_map, cost_map = {}, {}
                for sid in fit_ids:
                    value, vcost = stat(sid)
                    weak = value is None or value < t
                    confident = True if b is None else est_fit.get(sid, -np.inf) >= b
                    escalate = weak and confident
                    cell = (sid, strong) if escalate else (sid, cheap)
                    if cell not in rewards:
                        continue
                    acc_map[sid] = rewards[cell]
                    cost_map[sid] = (
                        costs.get((sid, cheap), 0.0)
                        + vcost
                        + (costs.get((sid, strong), 0.0) if escalate else 0.0)
                    )
                if not acc_map:
                    continue
                acc = float(np.mean(list(acc_map.values())))
                boot = sorted(
                    float(np.mean([cost_map[s] for s in sample if s in cost_map]))
                    for sample in resamples
                )
                if boot[int(0.8 * len(boot))] <= cap and (best_cfg is None or acc > best_cfg[2]):
                    best_cfg = (t, b, acc, float(np.mean(list(cost_map.values()))))
        variant = f"r2-vcascade-{verifier}-{version}" + ("-knn" if use_knn else "")
        if best_cfg is None:
            logger.info("%s/s%d %s: declined (no config under cap)", name, seed, variant)
            continue
        t, b, fit_acc, fit_cost = best_cfg

        def decide(
            sid: str,
            transcript: list,
            cheap: str = cheap,
            strong: str = strong,
            t: float = t,
            b: float | None = b,
            est_test: dict = est_test,
            scores: dict = scores,
        ) -> str | Finish:
            if not transcript:
                return cheap
            if len(transcript) == 1:
                rec = scores.get(str(episode_key(transcript[0])))
                weak = rec is None or rec["score"] is None or rec["score"] < t
                confident = True if b is None else est_test.get(sid, -np.inf) >= b
                return strong if (weak and confident) else Finish(pick=0)
            return Finish(pick=1)

        results = []
        extra_cost = []
        for permuted in (matrix, casc._swap_episodes(matrix)):
            result = evaluate_call_sequences(permuted, test_ids, decide, max_calls=2)
            # Charge the verifier call for the consumed cheap episode on every scenario.
            vsum = 0.0
            for sid in test_ids:
                _v, vcost = stat(sid)  # mean per-episode verifier cost for the cell
                vsum += vcost
            extra_cost.append(vsum / len(test_ids))
            results.append(result)
        mean_result = results[0].model_copy(
            update={
                "accuracy": float(np.mean([r.accuracy for r in results])),
                "cost_per_call": float(
                    np.mean([r.cost_per_call + e for r, e in zip(results, extra_cost, strict=True)])
                ),
            }
        )
        append_run(
            RunRecord(
                run_id=f"r2-{name}-iid-s{seed}-{variant}-{uuid.uuid4().hex[:8]}",
                ts=ts,
                matrix=name,
                variant=variant,
                params={
                    "family": "vcascade",
                    "verifier": verifier,
                    "prompt": version,
                    "knn": use_knn,
                    "split": "iid",
                },
                split_seed=seed,
                fit_scenarios=len(fit_ids),
                test_scenarios=len(test_ids),
                result=mean_result,
                baselines={"best_single": best_eval},
                notes=(
                    f"pair={cheap}->{strong} t={t} b={b} "
                    f"esc_rate={mean_result.model_mix.get(strong, 0.0):.2f} "
                    f"fit_acc={fit_acc:.4f} fit_cost={fit_cost:.5f} "
                    f"verifier_cost/call=${float(np.mean(extra_cost)):.5f} "
                    f"perm_accs={[round(r.accuracy, 4) for r in results]}"
                ),
            ),
            RUNS,
        )
        logger.info(
            "%s/s%d %s: acc=%.4f cost=$%.5f (best %.4f/$%.5f) t=%d esc=%.2f",
            name,
            seed,
            variant,
            mean_result.accuracy,
            mean_result.cost_per_call,
            best_eval.accuracy,
            best_eval.cost_per_call,
            t,
            mean_result.model_mix.get(strong, 0.0),
        )


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    version = "v1" if "--prompt=v1" in flags else "v0"
    seeds = [0, 1, 2, 3, 4]
    for flag in flags:
        if flag.startswith("--seeds="):
            seeds = [int(s) for s in flag.split("=", 1)[1].split(",")]
    command = args[0]
    if command == "score":
        matrix, _drv = _load_matrix(args[1])
        score_episodes(matrix, args[2], args[3], version=version)
    elif command == "eval":
        evaluate(
            args[1],
            args[2],
            args[3],
            args[4],
            version=version,
            use_knn="--knn" in flags,
            seeds=seeds,
        )
    else:
        raise SystemExit(f"unknown command {command}")


if __name__ == "__main__":
    main()
