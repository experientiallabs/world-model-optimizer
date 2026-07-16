"""Build the expanded, frozen eval pool (goal: generative synthesis > random; fair-eval leg).

The 21-scenario eval pool has a ~±5.5-pt draw-to-draw noise floor — too coarse to detect
selection/generation effects. This mines the FULL 211-trace eval split (hash-disjoint from
train) into a 50–80-scenario validated pool on the approved stack: gpt-5.4 synthesis (Foundry),
Opus 4.8 inline checklist validation (Bedrock us-east-2), enriched capability embeddings
(domain + tool signature). The pool is then FROZEN: committed, and every arm — baselines
included — is (re-)evaluated on it under identical protocol (k=3 means).

Usage (from the repo root, Mac creds):
    uv run python .agents/scripts/build_eval_pool_v2.py [--budget 95]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / ".agents" / "scripts"))

from collect_teacher import foundry, opus_judge  # noqa: E402
from run_scenario_e2e import TRACES, titan_embedder  # noqa: E402

from wmh.engine.build import split_traces  # noqa: E402
from wmh.ingest import get_adapter  # noqa: E402
from wmh.scenarios import ScenarioBuildConfig, build_scenario_set  # noqa: E402
from wmh.scenarios.mining.facets import TraceFacet, trace_domain  # noqa: E402

DISTILL = REPO / ".agents" / "docs" / "research" / "distill"
SYNTH_MODEL = "gpt-5.4"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--budget", type=int, default=95, help="synthesis budget (target 50-80 valid)"
    )
    parser.add_argument("--out", default="eval_pool_v2.json")
    args = parser.parse_args()
    t0 = time.time()

    traces = get_adapter("otel-genai").from_file(str(TRACES))
    _, eval_traces = split_traces(traces, 0.8)
    facet_data = json.loads((DISTILL / "facets_full.json").read_text())["eval"]
    facets = [TraceFacet.model_validate(f) for f in facet_data]
    assert len(facets) == len(eval_traces), "cached eval facets misaligned with eval split"
    # Cached facets predate the domain field; backfill so enriched embed_text has it.
    facets = [
        facet.model_copy(update={"domain": trace_domain(trace)})
        for facet, trace in zip(facets, eval_traces, strict=True)
    ]
    log.info(
        "eval split: %d traces, domains %s",
        len(eval_traces),
        dict(Counter(f.domain or "?" for f in facets)),
    )

    pool = build_scenario_set(
        eval_traces,
        facets,
        foundry(SYNTH_MODEL),
        titan_embedder(),
        ScenarioBuildConfig(budget=args.budget, seed=0),
        judge_provider=opus_judge(),
    )
    out = DISTILL / args.out
    pool.save(out)
    domains = Counter()
    by_id = {t.trace_id: t for t in eval_traces}
    for s in pool.scenarios:
        domains[trace_domain(by_id[s.provenance[0]]) or "?"] += 1
    log.info(
        "eval_pool_v2: %d valid scenarios, %d clusters, domains %s -> %s (%.0fs)",
        len(pool.scenarios),
        len({s.cluster_name for s in pool.scenarios}),
        dict(domains),
        out,
        time.time() - t0,
    )


if __name__ == "__main__":
    main()
