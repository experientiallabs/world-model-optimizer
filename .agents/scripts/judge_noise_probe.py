"""Judge-noise decomposition probe: how much episode-to-episode reward variance is the judge?

Gates the cascade / best-of-n / distilled-verifier families (drawing-board survey, findings/
master.md 2026-07-24): the measured best-of-2 bound (tau-bench +7.1pt at 3.6x cheaper) is only
real if max(r1, r2) picks a genuinely better ROLLOUT rather than a luckier JUDGE draw. The stored
matrices can't answer this — sessions are freed on close, so an episode can't be re-judged after
the fact. This probe runs fresh episodes and judges each session K times before it closes
(byte-identical judge input: `_build_reward_prompt` renders task + actions + observation content,
none of which the first judge call mutates), giving a direct measurement of within-session judge
spread to compare against the across-episode spread already in the matrix.

Reads: within-session judge SD (this probe) vs across-episode mean |r1-r2| (the stored matrix).
Judge spread << episode spread -> variance is rollout-driven and the bound stands.

Usage: uv run python .agents/scripts/judge_noise_probe.py [corpus ...]  (default: tau-bench)
"""

from __future__ import annotations

import json
import logging
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from wmo.config import load_config
from wmo.engine.world_model import WorldModel
from wmo.env.base import WorldModelEnv
from wmo.env.closed_loop import scenario_id
from wmo.env.episode import run_episode
from wmo.env.llm_agent import LLMAgent
from wmo.env.scenarios import scenarios_from_traces, tools_hint_from_traces
from wmo.ingest import get_adapter
from wmo.optimize.reward import EpisodeScore
from wmo.providers.base import ProviderConfig, ProviderKind
from wmo.providers.pool import load_pool, pool_provider
from wmo.providers.registry import get_provider

logging.basicConfig(level=logging.INFO, format="%(message)s")
for noisy in ("httpx", "urllib3", "botocore", "anthropic", "openai"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("judge_noise")

BUNDLES = Path("packages/environment-capture")
OUT = Path(".wmo/evals/judge_noise")
# Same sampling as run_wm_matrices.py so probe scenarios == matrix scenarios.
SCENARIOS_PER_CORPUS = 25
MAX_STEPS = 8
SEED = 5
JUDGE_REPEATS = 3
# Span the disagreement range seen in the matrix: glm-5.2 (most volatile cells), kimi-k2.6
# (the best-of-2 winner whose bound this gates), fable-5 (best-single anchor).
MODELS = ["glm-5.2", "kimi-k2.6", "fable-5"]

JUDGE = ProviderConfig(
    kind=ProviderKind.BEDROCK, model="us.anthropic.claude-opus-4-8", region="us-east-1"
)

# corpus -> (bundle, telecom filter) mirroring run_wm_matrices.py CORPORA.
CORPORA: dict[str, tuple[str, bool | None]] = {
    "tau-bench": ("tau-bench", False),
    "financebench": ("financebench", None),
    "continual-learning": ("continual-learning", None),
}


class RepeatJudgeEnv(WorldModelEnv):
    """WorldModelEnv that judges the session K times before ending it.

    The base env frees the session inside `close`, so repeat judging must happen there too.
    Individual judge failures are kept as None rather than aborting the episode: a throttled
    second call must not discard a measured first one.
    """

    def __init__(self, world_model: WorldModel, repeats: int) -> None:
        super().__init__(world_model, score_on_close=False)
        self._repeats = repeats
        self.scores: list[EpisodeScore | None] = []

    def close(self) -> None:
        if self._session_id is None:
            return
        self.scores = []
        for _ in range(self._repeats):
            try:
                self.scores.append(self._world_model.score_session(self._session_id))
            except Exception as exc:  # noqa: BLE001 - keep partial measurements
                logger.warning("judge call failed: %s", str(exc)[:120])
                self.scores.append(None)
        super().close()


def run_corpus(corpus: str) -> None:
    bundle, telecom = CORPORA[corpus]
    OUT.mkdir(parents=True, exist_ok=True)
    rows_path = OUT / f"{corpus}_rows.jsonl"
    done: set[tuple[str, str]] = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            row = json.loads(line)
            done.add((row["scenario_id"], row["model"]))
        logger.info("%s: resuming, %d rows captured", corpus, len(done))

    model_dir = BUNDLES / bundle / "models" / corpus
    traces = get_adapter("otel-genai").from_file(str(BUNDLES / bundle / "traces.otel.jsonl"))
    if telecom is not None:
        traces = [t for t in traces if ("telecom" in str(t.metadata).lower()) == telecom]
    scenarios = scenarios_from_traces(traces)
    rng = random.Random(SEED)
    if len(scenarios) > SCENARIOS_PER_CORPUS:
        scenarios = rng.sample(scenarios, SCENARIOS_PER_CORPUS)
    hint = tools_hint_from_traces(traces)

    pool = load_pool()
    serve_config = load_config(str(model_dir)).serve_provider_config()
    judge_provider = get_provider(JUDGE)
    write_lock = Lock()
    rows_handle = rows_path.open("a", encoding="utf-8")
    started = time.monotonic()

    def run_model(entry_name: str) -> None:
        wm = WorldModel.load(
            str(model_dir), get_provider(serve_config), reward_provider=judge_provider
        )
        entry = pool.entry(entry_name)
        for scenario in scenarios:
            sid = scenario_id(scenario)
            if (sid, entry_name) in done:
                continue
            env = RepeatJudgeEnv(wm, repeats=JUDGE_REPEATS)
            agent = LLMAgent(pool_provider(entry), temperature=0.0, tools_hint=hint or None)
            result = run_episode(env, agent, scenario.task, max_steps=MAX_STEPS)
            row = {
                "scenario_id": sid,
                "model": entry_name,
                "rewards": [s.reward if s else None for s in env.scores],
                "critiques": [s.critique if s else None for s in env.scores],
                "steps": len(result.steps),
                "stop_reason": str(result.stop_reason),
                "error": result.error,
            }
            with write_lock:
                rows_handle.write(json.dumps(row) + "\n")
                rows_handle.flush()

    with ThreadPoolExecutor(max_workers=len(MODELS)) as executor:
        futures = {executor.submit(run_model, name): name for name in MODELS}
        for future, name in futures.items():
            try:
                future.result()
                logger.info("%s: %s done", corpus, name)
            except Exception as exc:  # noqa: BLE001 - one model must not kill the probe
                logger.error("%s: %s FAILED: %s", corpus, name, str(exc)[:200])
    rows_handle.close()
    report(corpus)
    logger.info("%s: probe finished in %.0fs", corpus, time.monotonic() - started)


def report(corpus: str) -> None:
    """Within-session judge spread (probe) vs across-episode spread (stored matrix)."""
    rows_path = OUT / f"{corpus}_rows.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    spreads: list[float] = []
    per_model: dict[str, list[float]] = {}
    for row in rows:
        rewards = [r for r in row["rewards"] if r is not None]
        if len(rewards) < 2:
            continue
        spread = max(rewards) - min(rewards)
        spreads.append(spread)
        per_model.setdefault(row["model"], []).append(spread)
    if not spreads:
        logger.info("%s: no multi-judge rows yet", corpus)
        return
    matrix_path = Path(".wmo/evals/wm") / f"{corpus}_matrix.json"
    episode_deltas: list[float] = []
    if matrix_path.exists():
        matrix = json.loads(matrix_path.read_text())
        cells: dict[tuple[str, str], list[float]] = {}
        for o in matrix["outcomes"]:
            if o["reward"] is not None:
                cells.setdefault((o["scenario_id"], o["model"]), []).append(o["reward"])
        episode_deltas = [abs(rs[0] - rs[1]) for rs in cells.values() if len(rs) == 2]
    logger.info(
        "%s: JUDGE within-session spread mean=%.3f p90=%.3f zero=%d/%d  |  "
        "EPISODE across-rollout mean|d|=%.3f (n=%d)",
        corpus,
        statistics.mean(spreads),
        sorted(spreads)[int(0.9 * (len(spreads) - 1))],
        sum(1 for s in spreads if s == 0.0),
        len(spreads),
        statistics.mean(episode_deltas) if episode_deltas else float("nan"),
        len(episode_deltas),
    )
    for name, values in sorted(per_model.items()):
        logger.info(
            "  %s: judge spread mean=%.3f max=%.3f (n=%d)",
            name,
            statistics.mean(values),
            max(values),
            len(values),
        )


def main() -> None:
    wanted = sys.argv[1:] or ["tau-bench"]
    for corpus in wanted:
        if corpus not in CORPORA:
            raise SystemExit(f"unknown corpus {corpus!r}; known: {sorted(CORPORA)}")
        run_corpus(corpus)


if __name__ == "__main__":
    main()
