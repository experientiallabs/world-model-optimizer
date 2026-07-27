"""Stage C: closed-loop outcome matrices for the 10 committed base world models.

Per corpus: load the SHIPPED artifact (its own pinned serve provider, judge overridden to the
pinned Opus 4.8), sample scenarios from the bundle's traces (telecom metadata split for
tau-bench vs tau-telecom), derive the trace tool surface, and run the 9-model pool closed-loop
(2 episodes/scenario, max_steps 8, 9 model-workers in parallel, each with its own WorldModel
instance). One OutcomeMatrix JSON per corpus; a corpus with an existing matrix is skipped, so
the sweep resumes.

Usage: uv run python .agents/scripts/run_wm_matrices.py [corpus ...]  (default: all 10)
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from wmo.config import load_config
from wmo.engine.world_model import WorldModel
from wmo.env.base import WorldModelEnv
from wmo.env.closed_loop import evaluate_pool
from wmo.env.scenarios import scenarios_from_traces, tools_hint_from_traces
from wmo.ingest import get_adapter
from wmo.optimize.outcomes import OutcomeMatrix
from wmo.providers.base import ProviderConfig, ProviderKind
from wmo.providers.pool import ModelPool, load_pool
from wmo.providers.registry import get_provider

logging.basicConfig(level=logging.INFO, format="%(message)s")
for noisy in ("httpx", "urllib3", "botocore", "anthropic", "openai", "wmo.env.closed_loop"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("wm_matrices")

BUNDLES = Path("packages/environment-capture")
OUT = Path(".wmo/evals/wm")
SCENARIOS_PER_CORPUS = 25
EPISODES = 2
MAX_STEPS = 8
SEED = 5

JUDGE = ProviderConfig(
    kind=ProviderKind.BEDROCK, model="us.anthropic.claude-opus-4-8", region="us-east-1"
)

# (model name, bundle, telecom filter: None | True | False)
CORPORA: list[tuple[str, str, bool | None]] = [
    ("bird-sql", "bird-sql", None),
    ("terminal-tasks", "terminal-tasks", None),
    ("tau-bench", "tau-bench", False),
    ("tau-telecom", "tau-bench", True),
    ("continual-learning", "continual-learning", None),
    ("crmarena", "crmarena", None),
    ("dabstep", "dabstep", None),
    ("financebench", "financebench", None),
    ("gaia2", "gaia2", None),
    ("swe-bench", "swe-bench", None),
]


def run_corpus(name: str, bundle: str, telecom: bool | None) -> None:
    out_path = OUT / f"{name}_matrix.json"
    if out_path.exists():
        logger.info("%s: matrix exists, skipping", name)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    rows_path = OUT / f"{name}_rows.jsonl"
    done: set[tuple[str, str, int]] = set()
    prior = []
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            o = json.loads(line)
            done.add((o["scenario_id"], o["model"], o["episode"]))
            prior.append(o)
        logger.info("%s: resuming, %d rows already captured", name, len(done))
    import threading
    write_lock = threading.Lock()
    rows_handle = rows_path.open("a", encoding="utf-8")
    started = time.monotonic()
    model_dir = BUNDLES / bundle / "models" / name
    traces = get_adapter("otel-genai").from_file(str(BUNDLES / bundle / "traces.otel.jsonl"))
    if telecom is not None:
        traces = [t for t in traces if ("telecom" in str(t.metadata).lower()) == telecom]
    scenarios = scenarios_from_traces(traces)
    rng = random.Random(SEED)
    if len(scenarios) > SCENARIOS_PER_CORPUS:
        scenarios = rng.sample(scenarios, SCENARIOS_PER_CORPUS)
    hint = tools_hint_from_traces(traces)
    logger.info(
        "%s: %d traces -> %d scenarios (sampled %d), tools_hint=%d chars",
        name, len(traces), len(scenarios_from_traces(traces)), len(scenarios), len(hint),
    )

    pool = load_pool()
    config = load_config(str(model_dir))
    serve_config = config.serve_provider_config()
    judge_provider = get_provider(JUDGE)

    def run_candidate(entry_name: str) -> OutcomeMatrix:
        # Each worker gets its OWN WorldModel instance: sessions and retrieval buffers are
        # per-instance, so workers never share mutable env state.
        wm = WorldModel.load(
            str(model_dir), get_provider(serve_config), reward_provider=judge_provider
        )
        single = ModelPool(models=[pool.entry(entry_name)])
        from wmo.env.closed_loop import scenario_id as sid_of

        todo = [
            s
            for s in scenarios
            if any((sid_of(s), entry_name, ep) not in done for ep in range(EPISODES))
        ]

        def persist(outcome) -> None:  # noqa: ANN001
            with write_lock:
                rows_handle.write(outcome.model_dump_json() + "\n")
                rows_handle.flush()

        return evaluate_pool(
            lambda: WorldModelEnv(wm, score_on_close=True),
            single,
            todo,
            episodes_per_scenario=EPISODES,
            max_steps=MAX_STEPS,
            tools_hint=hint or None,
            on_outcome=persist,
        )

    from wmo.optimize.outcomes import ScenarioOutcome

    outcomes = [ScenarioOutcome.model_validate(o) for o in prior]
    with ThreadPoolExecutor(max_workers=len(pool.models)) as executor:
        futures = {
            executor.submit(run_candidate, entry.name): entry.name for entry in pool.models
        }
        for future, entry_name in futures.items():
            try:
                outcomes.extend(future.result().outcomes)
                logger.info("%s: %s done", name, entry_name)
            except Exception as exc:  # noqa: BLE001 - one candidate must not kill the corpus
                logger.error("%s: %s FAILED: %s", name, entry_name, str(exc)[:200])

    # Dedupe (scenario, model, episode): a resumed scenario reruns all its episodes, so a
    # previously-persisted episode can appear twice; first occurrence wins.
    seen: set[tuple[str, str, int]] = set()
    unique = []
    for outcome in outcomes:
        key = (outcome.scenario_id, outcome.model, outcome.episode)
        if key in seen:
            continue
        seen.add(key)
        unique.append(outcome)
    matrix = OutcomeMatrix(pool=pool.models, outcomes=unique)
    matrix.save(out_path)
    scored = [o for o in matrix.outcomes if o.reward is not None]
    candidate_cost = sum(o.cost_usd for o in matrix.outcomes)
    logger.info(
        "%s: %d outcomes (%d scored) candidate-cost $%.2f in %.0fs -> %s",
        name, len(matrix.outcomes), len(scored), candidate_cost,
        time.monotonic() - started, out_path,
    )


def main() -> None:
    wanted = sys.argv[1:] or [name for name, _b, _t in CORPORA]
    for name, bundle, telecom in CORPORA:
        if name in wanted:
            run_corpus(name, bundle, telecom)
    logger.info("all done: %s", json.dumps(sorted(p.name for p in OUT.glob("*_matrix.json"))))


if __name__ == "__main__":
    main()
