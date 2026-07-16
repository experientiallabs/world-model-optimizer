"""Judge real-env rollout transcripts with the same checklist judge as the WM evals.

Phase 2 of the sim2real leg: consumes real_env_rollout.py JSONL, grades each episode with Gemini
Flash against the scenario checklist, writes a summary comparable to the WM eval JSONs.

Usage (from the repo root):
    uv run python .agents/scripts/judge_real_transcripts.py \
        --transcripts .agents/docs/research/distill/real_rollouts_before.jsonl \
        --label real-student-before
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / ".agents" / "scripts"))

from collect_teacher import _load_gemini_key, gemini  # noqa: E402

from wmh.core.types import Action, ActionKind, Observation, Step  # noqa: E402
from wmh.scenarios import ChecklistJudge  # noqa: E402

DISTILL = REPO / ".agents" / "docs" / "research" / "distill"


def to_steps(record: dict) -> list[Step]:
    def as_dict(value):  # noqa: ANN001, ANN202 - agents sometimes emit positional-arg lists
        return value if isinstance(value, dict) else {"args": value}

    steps = [
        Step(
            action=Action(
                kind=ActionKind.TOOL_CALL,
                name=s["tool"],
                arguments=as_dict(s.get("arguments") or {}),
            ),
            observation=Observation(content=s["observation"], is_error=s.get("is_error", False)),
        )
        for s in record["steps"]
    ]
    if record.get("final_text"):
        steps.append(
            Step(
                action=Action(kind=ActionKind.MESSAGE, content=record["final_text"]),
                observation=Observation(content=""),
            )
        )
    return steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcripts", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    _load_gemini_key()
    judge = ChecklistJudge(gemini("gemini-2.5-flash"))
    records = [
        json.loads(line)
        for line in Path(args.transcripts).read_text(encoding="utf-8").splitlines()
        if line
    ]
    t0 = time.time()

    def grade(record: dict) -> dict:
        verdict = judge.score(record["task"], record["checklist"], to_steps(record))
        return {
            "scenario_id": record["scenario_id"],
            "domain": record["domain"],
            "pass_index": record["pass_index"],
            "success": verdict.success,
            "pass_rate": verdict.pass_rate,
            "steps": len(record["steps"]),
        }

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(grade, records))

    by_domain: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_domain[r["domain"]].append(r)
    summary = {
        "label": args.label,
        "episodes": len(results),
        "success_rate": sum(r["success"] for r in results) / max(1, len(results)),
        "mean_pass_rate": sum(r["pass_rate"] for r in results) / max(1, len(results)),
        "per_domain": {
            d: {
                "episodes": len(rs),
                "success_rate": sum(r["success"] for r in rs) / len(rs),
            }
            for d, rs in sorted(by_domain.items())
        },
        "wall_clock_seconds": round(time.time() - t0, 1),
        "results": results,
    }
    out = DISTILL / f"eval_{args.label}.json"
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(
        f"{args.label}: success {summary['success_rate']:.1%} over {len(results)} real-env "
        f"episodes -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
