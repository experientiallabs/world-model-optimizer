"""OVN-D: truth-scored financebench mini-leg - answer vs PINNED GOLD, no wm judge.

Live closed-loop episodes against the financebench world model with the D-COMPRESS
stage applied per arm, scored by a DIRECT deterministic gold scorer (the wm judge's
three documented bugs disqualify it; DECISIONS 2026-07-27). Tasks come from the bundle's
data/{test,train}.jsonl, gold from gold/<task_id>.json (answer text + pinned numeric).

Scorer: extract scale-worded numbers from the agent's FINAL reply (million/billion/
thousand suffixes honored, bare numbers taken at face value), correct if any extracted
value matches gold numeric within 2% relative tolerance; a no-answer episode scores 0
and is flagged unanswered. Every (reply tail, gold, extracted, verdict) is logged to the
rows JSONL for morning eyeballing; the scorer is deliberately strict and simple, and its
misses are auditable.

Arms are run one at a time (this runner executes ONE arm per invocation, so achieved
live ratios from the method arms can set the control aggressiveness afterwards):

    uv run python .agents/scripts/run_truth_mini_leg.py --arm off
    uv run python .agents/scripts/run_truth_mini_leg.py --arm aa            # A/A: byte-identical to off
    uv run python .agents/scripts/run_truth_mini_leg.py --arm stock --aggressiveness 0.5
    uv run python .agents/scripts/run_truth_mini_leg.py --arm adapted --aggressiveness <matched>
    uv run python .agents/scripts/run_truth_mini_leg.py --arm control-truncate --aggressiveness <measured>
    uv run python .agents/scripts/run_truth_mini_leg.py --arm control-random --aggressiveness <measured>

Metering: every episode's candidate + env token spend appended to the rows; cumulative
into cache/metering-c1.jsonl per invocation.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from datetime import UTC, datetime
from pathlib import Path

from wmo.config import load_config
from wmo.engine.world_model import WorldModel
from wmo.env.episode import run_episode
from wmo.env.llm_agent import LLMAgent
from wmo.env.scenarios import tools_hint_from_traces
from wmo.ingest import get_adapter
from wmo.env.base import WorldModelEnv
from wmo.env.closed_loop import _CompressingProvider, _TimedProvider
from wmo.optimize.compression import CompressionConfig, register_compressor
from wmo.providers.pool import load_pool, pool_provider
from wmo.providers.registry import get_provider


log = logging.getLogger("truth_mini_leg")

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path.home() / "Desktop/Projects/wmh-compression-data"
BUNDLE = REPO / "packages/environment-capture/financebench"
MODEL_DIR = BUNDLE / "models/financebench"
OUT_DIR = DATA_ROOT / "matrices"
METERING_PATH = DATA_ROOT / "cache/metering-c1.jsonl"

MODELS = ["gpt-5.4-mini", "sonnet-5"]
N_TRAIN_TASKS = 15
EPISODES = 2  # default; --episodes overrides
MAX_STEPS = 16
SEED = 0

NUM_RE = re.compile(
    r"(-?\$?\d[\d,]*\.?\d*)\s*(million|billion|thousand|mn|bn|k\b|m\b|b\b)?", re.IGNORECASE
)
SCALES = {
    "million": 1e6,
    "mn": 1e6,
    "m": 1e6,
    "billion": 1e9,
    "bn": 1e9,
    "b": 1e9,
    "thousand": 1e3,
    "k": 1e3,
}


def extract_numbers(text: str) -> list[float]:
    out: list[float] = []
    for raw, scale in NUM_RE.findall(text):
        try:
            value = float(raw.replace(",", "").replace("$", ""))
        except ValueError:
            continue
        out.append(value * SCALES.get((scale or "").lower(), 1.0))
    return out


GOLD_SCALES = (1e-3, 1.0, 1e3, 1e6, 1e9)


def year_like(value: float) -> bool:
    """Gold numerics that are bare years match any date mention; they cannot score."""
    return float(value).is_integer() and 1900 <= value <= 2100


def gold_correct(reply: str, gold_numeric: float, tolerance: float = 0.02) -> bool:
    """Two-sided decade-scale matching: gold in millions notation vs reply in raw
    dollars (and vice versa) both count; anything else does not."""
    if gold_numeric == 0:
        return any(abs(v) < 1e-9 for v in extract_numbers(reply))
    for v in extract_numbers(reply):
        for s in GOLD_SCALES:
            target = gold_numeric * s
            if target and abs(v - target) / abs(target) <= tolerance:
                return True
    return False


ALL_TASKS = False


def load_tasks() -> list[dict]:
    tasks = []
    for split in ("test", "train"):
        for ln in (BUNDLE / f"data/{split}.jsonl").open():
            row = json.loads(ln)
            gold_path = BUNDLE / f"gold/{row['task_id']}.json"
            if not gold_path.exists():
                continue
            gold = json.loads(gold_path.read_text())
            if gold.get("numeric") is None:
                continue
            if year_like(gold["numeric"]):
                log.info("excluding %s: year-like gold %.0f cannot score", row["task_id"], gold["numeric"])
                continue
            tasks.append(
                {
                    "task_id": row["task_id"],
                    "prompt": row["prompt"],
                    "stratum": split,
                    "gold_answer": gold["answer"],
                    "gold_numeric": gold["numeric"],
                }
            )
    test = [t for t in tasks if t["stratum"] == "test"]
    train = [t for t in tasks if t["stratum"] == "train"]
    rng = random.Random(SEED)
    rng.shuffle(train)
    picked = test + (train if ALL_TASKS else train[:N_TRAIN_TASKS])
    log.info("tasks: %d test + %d train sampled (seed %d)", len(test), len(picked) - len(test), SEED)
    return picked


class RandomRemovalCompressor:
    """Matched-ratio random word removal control (seeded per segment: deterministic and
    segment-local, so unchanged segments always emit identical bytes = append-stable at
    the seam's segment grain)."""

    id = "random-removal"
    version = "1"
    append_stable = True

    def compress(self, segments: list[str], config: CompressionConfig):  # noqa: ANN201
        import hashlib
        from wmo.optimize.compression import CompressionResult, estimate_tokens

        start = time.monotonic()
        word_re = re.compile(r"\S+\s*")
        out: list[str] = []
        for segment in segments:
            if config.aggressiveness == 0.0:
                out.append(segment)
                continue
            words = word_re.findall(segment)
            seed = int.from_bytes(hashlib.sha256(segment.encode()).digest()[:8], "big")
            rng = random.Random(seed)
            n_remove = round(len(words) * config.aggressiveness)
            drop = set(rng.sample(range(len(words)), min(n_remove, len(words))))
            out.append("".join(w for i, w in enumerate(words) if i not in drop))
        return CompressionResult(
            segments=out,
            tokens_in_raw=sum(estimate_tokens(s) for s in segments),
            tokens_in_compressed=sum(estimate_tokens(s) for s in out),
            latency_s=time.monotonic() - start,
        )


class AdaptedEndpointStyleCompressor:
    """The adapted checkpoint behind the CANONICAL scorer class, in process (CPU fp32).

    Loads deploy/compressor-endpoint/server.py's LLMLingua2FixedThreshold with the
    adapted checkpoint dir as model_id, so packing and selection are byte-canonical.
    Reads `aggressiveness` as the absolute keep threshold, exactly like the endpoint
    client. append_stable holds for the same reason (per-word local, fixed bar).
    """

    id = "adapted-financebench-inprocess"  # per-process registry; arm name distinguishes runs
    version = "r2-full-ft"
    append_stable = True

    def __init__(self, checkpoint: str) -> None:
        import hashlib
        import importlib.util
        import sys

        path = REPO / "deploy/compressor-endpoint/server.py"
        self.server_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        spec = importlib.util.spec_from_file_location("compressor_server", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["compressor_server"] = mod
        spec.loader.exec_module(mod)
        self._inner = mod.LLMLingua2FixedThreshold(checkpoint, "cpu")

    def compress(self, segments: list[str], config: CompressionConfig):  # noqa: ANN201
        from wmo.optimize.compression import CompressionResult, estimate_tokens

        start = time.monotonic()
        if config.aggressiveness == 0.0:
            raw = sum(estimate_tokens(s) for s in segments)
            return CompressionResult(
                segments=list(segments), tokens_in_raw=raw, tokens_in_compressed=raw,
                latency_s=time.monotonic() - start,
            )
        outcome = self._inner.compress(segments, config.aggressiveness)
        return CompressionResult(
            segments=list(outcome.segments),
            tokens_in_raw=sum(estimate_tokens(s) for s in segments),
            tokens_in_compressed=sum(estimate_tokens(s) for s in outcome.segments),
            latency_s=time.monotonic() - start,
        )


def build_compression(arm: str, aggressiveness: float, checkpoint: str | None) -> CompressionConfig | None:
    if arm in ("off", "aa"):
        return None
    if arm == "stock":
        import wmo.optimize.compression_endpoint  # noqa: F401 - registers the factory

        return CompressionConfig(
            compressor_id="llmlingua2-endpoint", aggressiveness=aggressiveness
        )
    if arm.startswith("adapted"):
        register_compressor(AdaptedEndpointStyleCompressor(checkpoint))
        return CompressionConfig(
            compressor_id="adapted-financebench-inprocess", aggressiveness=aggressiveness
        )
    if arm == "control-truncate":
        return CompressionConfig(compressor_id="truncate", aggressiveness=aggressiveness)
    if arm == "control-random":
        register_compressor(RandomRemovalCompressor())
        return CompressionConfig(compressor_id="random-removal", aggressiveness=aggressiveness)
    raise SystemExit(f"unknown arm {arm}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "urllib3", "botocore", "anthropic", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    global ALL_TASKS, EPISODES
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--aggressiveness", type=float, default=0.0)
    ap.add_argument("--checkpoint", default=str(Path.home() / "Desktop/Projects/wmh-compression-data/cache/adapted-financebench-full"))
    ap.add_argument("--cap-usd", type=float, default=12.0, help="candidate+env metered stop per invocation")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--all-tasks", action="store_true", help="all scorable golds, not the 18-task mini set")
    ap.add_argument("--capture-segments", action="store_true", help="dump raw live user segments per episode (off arm)")
    ap.add_argument("--rows-prefix", default="truth-mini")
    args = ap.parse_args()
    EPISODES = args.episodes

    ALL_TASKS = args.all_tasks
    tasks = load_tasks()
    traces = get_adapter("otel-genai").from_file(str(BUNDLE / "traces.otel.jsonl"))
    tools_hint = tools_hint_from_traces(traces)
    log.info("tools_hint: %d chars", len(tools_hint))
    compression = build_compression(args.arm, args.aggressiveness, args.checkpoint)
    pool = load_pool()
    config = load_config(str(MODEL_DIR))
    serve_config = config.serve_provider_config()
    rows_path = OUT_DIR / f"{args.rows_prefix}-{args.arm}_rows.jsonl"
    capture_path = OUT_DIR / f"{args.rows_prefix}-{args.arm}_segments.jsonl"
    done: set[tuple[str, str, int]] = set()
    if rows_path.exists():
        done = {
            (r["task_id"], r["model"], r["episode"])
            for r in map(json.loads, rows_path.open())
        }
        log.info("resuming: %d rows exist", len(done))
    spent = 0.0
    n_correct = n_total = 0
    import threading
    from concurrent.futures import ThreadPoolExecutor

    lock = threading.Lock()
    stop = threading.Event()
    with rows_path.open("a") as f:
        for model_name in args.models.split(","):
            entry = pool.entry(model_name)
            jobs = [
                (task, episode)
                for task in tasks
                for episode in range(EPISODES)
                if (task["task_id"], model_name, episode) not in done
            ]

            def run_one(job):  # noqa: ANN001, ANN202
                nonlocal spent, n_correct, n_total
                if stop.is_set():
                    return
                task, episode = job
                try:
                    self_contained(task, episode)
                except Exception as exc:  # noqa: BLE001 - one broken episode never kills the arm
                    log.warning("episode error %s ep%d: %s", task["task_id"], episode, str(exc)[:150])

            def self_contained(task, episode):  # noqa: ANN001, ANN202
                nonlocal spent, n_correct, n_total
                timed = _TimedProvider(pool_provider(entry))
                compressing = (
                    _CompressingProvider(timed, compression)
                    if compression is not None
                    else None
                )
                agent = LLMAgent(compressing or timed, temperature=0.0, tools_hint=tools_hint)
                wm = WorldModel.load(str(MODEL_DIR), get_provider(serve_config))
                env = WorldModelEnv(wm, score_on_close=False)
                framed = (
                    task["prompt"]
                    + "\n\nThe source financial documents are files under docs/ in this "
                    "environment; list and read them with the bash tool (ls docs, grep, cat) "
                    "and answer from what they contain."
                )
                result = run_episode(env, agent, framed, max_steps=MAX_STEPS)
                if args.capture_segments:
                    observations = [
                        s.observation.content
                        for s in result.steps
                        if s.observation is not None and s.observation.content
                    ]
                    with lock:
                        with capture_path.open("a") as cf:
                            cf.write(
                                json.dumps(
                                    {
                                        "task_id": task["task_id"],
                                        "model": model_name,
                                        "episode": episode,
                                        "segments": [framed] + [str(x) for x in observations],
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                final = timed.replies[-1] if timed.replies else ""
                correct = gold_correct(final, task["gold_numeric"])
                answered = bool(extract_numbers(final))
                candidate_cost = entry.cost_usd(timed.usage)
                with lock:
                    spent += candidate_cost + 0.0685  # env at the master-measured rate
                    n_total += 1
                    n_correct += int(correct)
                    if spent > args.cap_usd:
                        stop.set()
                    f.write(
                        json.dumps(
                        {
                            "ts": datetime.now(UTC).isoformat(),
                    "owner": "c1",
                            "arm": args.arm,
                            "aggressiveness": args.aggressiveness,
                            "task_id": task["task_id"],
                            "stratum": task["stratum"],
                            "model": model_name,
                            "episode": episode,
                            "correct": correct,
                            "answered": answered,
                            "gold_numeric": task["gold_numeric"],
                            "extracted": extract_numbers(final)[:6],
                            "final_reply_tail": final[-400:],
                            "steps": len(result.steps),
                            "stop_reason": str(result.stop_reason),
                            "candidate_cost_usd": round(candidate_cost, 5),
                            "tokens_in_raw": compressing.tokens_in_raw if compressing else 0,
                            "tokens_in_compressed": compressing.tokens_in_compressed
                            if compressing
                            else 0,
                            "error": result.error,
                        },
                        ensure_ascii=False,
                    )
                        + "\n"
                    )
                    f.flush()

            with ThreadPoolExecutor(max_workers=4) as pool_exec:
                list(pool_exec.map(run_one, jobs))
            if stop.is_set():
                raise SystemExit(f"metered spend ${spent:.2f} exceeded cap; halting")
            log.info(
                "%s/%s: accuracy so far %d/%d, est spend $%.2f",
                args.arm,
                model_name,
                n_correct,
                n_total,
                spent,
            )
    with METERING_PATH.open("a") as f:
        f.write(
            json.dumps(
                {
                    "ts": datetime.now(UTC).isoformat(),
                    "provider_model": f"mini-leg:{args.arm}",
                    "spend_usd": round(spent, 2),
                    "note": "candidate metered exactly; env at measured $0.0685/ep flat",
                }
            )
            + "\n"
        )
    log.info("arm %s complete: %d/%d correct, est spend $%.2f", args.arm, n_correct, n_total, spent)


if __name__ == "__main__":
    main()
