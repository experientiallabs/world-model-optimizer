"""Pull SWE-bench Verified gold for the corpus instances into a committed cache.

The pinned scenarios carry a `rubric` populated from SWE-bench's GOLD evaluation criteria — the
per-instance `FAIL_TO_PASS` tests (the tests the correct patch must flip red->green; they ARE the
grading gold). Those live in the upstream dataset, not in our trace corpus, so this one-off pull
caches exactly the corpus's instances into `swe_gold.json`. `swe_pin_scenarios.py` then reads that
committed cache (stdlib only) and never needs the heavy `datasets`/`swebench` stack — the same
reason the trace corpus itself is a committed, reproducible artifact.

Which dataset: `SWE-bench/SWE-bench_Verified`, split `test` — the exact benchmark this example
captures (see ../README.md and ../run_real_scenario.py `--dataset` default; the overnight capture
runs mini-swe-agent with `--subset verified --split test`).

`FAIL_TO_PASS` / `PASS_TO_PASS` are JSON-encoded string lists in the dataset. The cache keeps
`fail_to_pass` verbatim (the gold), `pass_to_pass_count` as a count only (PASS_TO_PASS can be
hundreds of tests — the count is enough for the rubric), plus `repo` and `base_commit`.

Run in an ephemeral env so `datasets` never lands in the root pyproject (rule 5 / monorepo):
    uv run --with datasets python packages/environment-capture/swe-bench/rl/fetch_gold.py
Re-running on the same dataset + corpus rewrites a byte-identical cache.
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

from wmh.ingest import get_adapter

_HERE = Path(__file__).resolve().parent
_TRACES_PATH = _HERE.parent / "traces.otel.jsonl"
GOLD_OUT = _HERE / "swe_gold.json"
DATASET = "SWE-bench/SWE-bench_Verified"
SPLIT = "test"


def _json_list(value: object) -> list[str]:
    """Decode a SWE-bench FAIL_TO_PASS/PASS_TO_PASS field (JSON-encoded string list, or list)."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        decoded = json.loads(value)
        return [str(v) for v in decoded]
    raise TypeError(f"unexpected test-list field type: {type(value).__name__}")


def main() -> int:
    traces = get_adapter("otel-genai").from_file(str(_TRACES_PATH))
    corpus_ids = {
        t.metadata.get("instance_id")
        for t in traces
        if isinstance(t.metadata.get("instance_id"), str)
    }

    rows = load_dataset(DATASET, split=SPLIT)
    gold: dict[str, dict[str, object]] = {}
    for row in rows:
        iid = row["instance_id"]
        if iid not in corpus_ids:
            continue
        gold[iid] = {
            "repo": row["repo"],
            "base_commit": row["base_commit"],
            "fail_to_pass": sorted(_json_list(row["FAIL_TO_PASS"])),
            "pass_to_pass_count": len(_json_list(row["PASS_TO_PASS"])),
        }

    missing = sorted(corpus_ids - gold.keys())
    if missing:
        raise SystemExit(
            f"{len(missing)} corpus instance(s) not found in {DATASET}:{SPLIT}: {missing[:10]}"
            f"{' ...' if len(missing) > 10 else ''}"
        )

    GOLD_OUT.write_text(
        json.dumps(dict(sorted(gold.items())), indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"cached gold for {len(gold)}/{len(corpus_ids)} corpus instances -> {GOLD_OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
