#!/usr/bin/env python3
"""Evaluate a policy checkpoint on the REAL tau2-bench gym over the pinned eval scenarios.

This is the real-environment counterpart of the D30 WM eval (BENCH-B, D67): the SAME 20
pinned held-out scenarios, but the episode runs in Sierra's real tau2 environment (real
domain tools over the real JSON DB, real LLM user-simulator) and the verdict comes from
tau2's real grader — no world model, no judge. Records are keyed by the scenarios' source
trace ids so they pair 1:1 with the WM-eval records in
``.agents/docs/research/wm_tau_eval_results/*.jsonl``.

Split discipline: task selection is resolved FROM the pinned ``scenarios_eval.jsonl`` via
each scenario's provenance trace id -> the corpus trace's ``wmh.trace.metadata``
(domain, task_id). Nothing here derives its own split (D67).

Runs the ``tau2`` CLI via subprocess, so it needs the tau2 venv from
the tau-bench README on PATH (or ``--tau2-bin``) but imports nothing beyond
stdlib itself. The policy is any OpenAI-compatible endpoint (vLLM serve of a spliced
checkpoint); the user-simulator LLM is PINNED by default so every checkpoint row faces
the same simulated customers.

Usage (box with the tau2 venv + a vLLM serving the checkpoint on :8004):
    python real_eval.py \
        --agent-model hosted_vllm/rpp_n4_0192 \
        --agent-api-base http://127.0.0.1:8004/v1 \
        --tau2-bin  ~/world-model-harness/packages/environment-capture/tau-bench/.venv/bin/tau2 \
        --data-dir  ~/world-model-harness/packages/environment-capture/tau-bench/tau2-bench/data \
        --save-tag  rpp_n4_0192 \
        --out /data/output/real_tau_rpp_n4_0192.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
DEFAULT_SCENARIOS = _HERE / "scenarios_eval.jsonl"
DEFAULT_CORPUS = _HERE.parent / "traces.otel.jsonl"
# Pinned user simulator (D67): capture used Opus on Bedrock; eval keeps one fixed
# simulator across every checkpoint row so rows differ only by policy.
DEFAULT_USER_LLM = "bedrock/us.anthropic.claude-opus-4-8"

# The pinned telecom scenarios were captured from telecom's 2285-task "full" split
# (get_tasks("telecom", task_split_name="full") — see capture_telecom_multimodel.py);
# tau2's default split is "base" and raises on the missing ids. (--task-set-name
# telecom_full is NOT equivalent: that set's loader rejects split names entirely.)
TASK_SPLIT_OVERRIDES = {"telecom": "full"}


def resolve_tasks(scenarios_path: Path, corpus_path: Path) -> list[dict]:
    """Map each pinned eval scenario to its real tau2 (domain, task_id) via provenance."""
    scenarios = [
        json.loads(line) for line in scenarios_path.read_text().splitlines() if line.strip()
    ]
    wanted = {s["provenance"][0] for s in scenarios}
    resolved: dict[str, dict] = {}
    with corpus_path.open() as f:
        for line in f:
            span = json.loads(line)
            tid = span["traceId"]
            if tid not in wanted or tid in resolved:
                continue
            for attr in span.get("attributes", []):
                if attr["key"] == "wmh.trace.metadata":
                    md = json.loads(attr["value"]["stringValue"])
                    resolved[tid] = {
                        "trace_id": tid,
                        "domain": md["domain"],
                        "task_id": str(md["task_id"]),
                    }
            if len(resolved) == len(wanted):
                break  # the corpus is ~100MB; stop once every pinned id is resolved
    missing = wanted - set(resolved)
    if missing:
        raise SystemExit(f"unresolved provenance ids (corpus mismatch?): {sorted(missing)}")
    # by_task in main() keys records by (domain, task_id); two pinned scenarios sharing a
    # task_id would silently merge their rows and break the 1:1 WM-eval pairing.
    pairs = [(r["domain"], r["task_id"]) for r in resolved.values()]
    dupes = {pair for pair in pairs if pairs.count(pair) > 1}
    if dupes:
        raise SystemExit(
            f"pinned scenarios share (domain, task_id) — pairing ambiguous: {sorted(dupes)}"
        )
    return [resolved[s["provenance"][0]] | {"scenario_domain": s["domain"]} for s in scenarios]


def run_domain(args: argparse.Namespace, domain: str, task_ids: list[str], save_name: str) -> Path:
    """One ``tau2 run`` covering all of a domain's pinned tasks; returns results.json path."""
    cmd = [
        args.tau2_bin,
        "run",
        "--domain",
        domain,
        *(
            ["--task-split-name", TASK_SPLIT_OVERRIDES[domain]]
            if domain in TASK_SPLIT_OVERRIDES
            else []
        ),
        "--task-ids",
        *task_ids,
        "--num-trials",
        str(args.trials),
        "--max-concurrency",
        str(args.concurrency),
        "--agent-llm",
        args.agent_model,
        "--agent-llm-args",
        # temperature mirrors the pinned D30 WM-eval protocol (1.0) so real-env rows and
        # WM rows differ only by environment, not sampling.
        json.dumps(
            {"temperature": args.agent_temperature}
            | (
                {"api_base": args.agent_api_base, "api_key": args.agent_api_key}
                if args.agent_api_base
                else {}
            )
            | (json.loads(args.agent_llm_extra) if args.agent_llm_extra else {})
        ),
        "--user-llm",
        args.user_llm,
        "--user-llm-args",
        "{}",
        "--save-to",
        save_name,
        # Reruns must not block on the interactive "resume? (y/n)" prompt (observed: a
        # headless rerun sat on it for hours).
        "--auto-resume",
    ]
    env = os.environ | {"TAU2_DATA_DIR": args.data_dir}
    print(f"[real_eval] {domain}: {len(task_ids)} tasks x {args.trials} trials", flush=True)
    subprocess.run(cmd, check=True, env=env)
    return Path(args.data_dir) / "simulations" / save_name / "results.json"


def collect_records(results_path: Path, by_task: dict[str, str]) -> list[dict]:
    """Flatten a tau2 results.json into paired-analysis records (one per simulation)."""
    data = json.loads(results_path.read_text())
    trial_counter: dict[str, int] = defaultdict(int)
    records = []
    for sim in data.get("simulations", []) or []:
        task_id = str(sim.get("task_id", ""))
        trace_id = by_task.get(task_id)
        if trace_id is None:
            continue  # not one of ours (defensive; --task-ids should prevent this)
        reward = (sim.get("reward_info") or {}).get("reward")
        rollout = trial_counter[task_id]
        trial_counter[task_id] += 1
        records.append(
            {
                "scenario_id": trace_id,  # pairs with WM-eval records
                "rollout_index": rollout,
                "task_id": task_id,
                "reward": reward if reward is not None else 0.0,
                "success": reward is not None and reward >= 1.0 - 1e-9,
                "errors": [] if reward is not None else ["no reward_info in simulation"],
                "env": "real-tau2",
                "sim_id": str(sim.get("id", "")),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", default=str(DEFAULT_SCENARIOS))
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--agent-model", required=True, help="litellm spec for the policy")
    parser.add_argument("--agent-api-base", default=None)
    parser.add_argument("--agent-api-key", default="dummy")
    parser.add_argument("--agent-temperature", type=float, default=1.0)
    parser.add_argument(
        "--agent-llm-extra",
        default=None,
        help="JSON merged into --agent-llm-args (e.g. vLLM extra_body: think-free checkpoints "
        'need \'{"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}\' '
        "or they emit immediate EOS — the v1 SFT trap, D70/D30 notes).",
    )
    parser.add_argument("--user-llm", default=DEFAULT_USER_LLM)
    parser.add_argument("--trials", type=int, default=2, help="episodes per task (D30: 2)")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--tau2-bin", default="tau2")
    parser.add_argument("--data-dir", required=True, help="TAU2_DATA_DIR (the repo's data/)")
    parser.add_argument("--save-tag", required=True, help="simulation save-name prefix")
    parser.add_argument("--out", required=True, help="records jsonl for paired analysis")
    args = parser.parse_args()

    tasks = resolve_tasks(Path(args.scenarios), Path(args.corpus))
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for t in tasks:
        by_domain[t["domain"]].append(t)

    all_records: list[dict] = []
    failures: list[str] = []
    for domain, ts in sorted(by_domain.items()):
        by_task = {t["task_id"]: t["trace_id"] for t in ts}
        save_name = f"{args.save_tag}_{domain}"
        try:
            results_path = run_domain(args, domain, sorted(by_task), save_name)
            all_records.extend(collect_records(results_path, by_task))
        except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as exc:
            failures.append(f"{domain}: {type(exc).__name__}: {exc}")
            print(f"[real_eval] DOMAIN FAILED {domain}: {exc}", file=sys.stderr, flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    ok = [r for r in all_records if not r["errors"]]
    succ = sum(r["success"] for r in ok)
    print(
        f"[real_eval] REAL TAU RESULTS ({args.save_tag}): "
        f"{len(all_records)} records, success {succ}/{len(ok)} "
        f"({succ / max(len(ok), 1) * 100:.1f}%), "
        f"mean reward {sum(r['reward'] for r in ok) / max(len(ok), 1):.3f}, "
        f"domain failures: {failures or 'none'}",
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
