"""LLMRouterBench (arXiv 2601.07206) adapter: their outcome records as our `OutcomeMatrix`.

The modern routing benchmark (33 models x 21+ datasets, hard 2025-era tasks) whose framework
re-implements Avengers as a published baseline, giving our replication a direct band to land
in. Records live under `results/bench-release/<dataset>/[hybrid/]<model>/<run>.json`, one list
of {index, origin_query, prompt, score, cost, ...} per (dataset, model); scenarios join on
(dataset, index). Only scenarios covered by EVERY requested model are kept (logged, never
silent) so comparisons stay apples-to-apples.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.providers.base import ProviderKind
from wmo.providers.pool import PoolEntry

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# The performance-cost ("flagship") track roster, as named in the record tree.
FLAGSHIP_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "claude-sonnet-4",
    "qwen3-235b-a22b-2507",
    "qwen3-235b-a22b-thinking-2507",
    "gpt-5-chat",
    "gpt-5",
    "glm-4.6",
    "kimi-k2-0905",
    "deepseek-v3.1-terminus",
    "deepseek-v3-0324",
    "deepseek-r1-0528",
    "intern-s1",
]


def _model_dir(dataset_dir: Path, model: str) -> Path | None:
    for candidate in (dataset_dir / model, dataset_dir / "hybrid" / model):
        if candidate.is_dir():
            return candidate
    return None


def _load_records(model_dir: Path) -> dict[int, dict]:
    """Latest run file wins (files are timestamp-suffixed); records keyed by index."""
    files = sorted(model_dir.glob("*.json"))
    if not files:
        return {}
    data = json.loads(files[-1].read_text(encoding="utf-8"))
    # Two shipped shapes: a bare record list, or a summary dict wrapping "records".
    records = data["records"] if isinstance(data, dict) else data
    return {int(rec["index"]): rec for rec in records}


def load_llmrouterbench(
    root: Path,
    *,
    models: list[str] | None = None,
    datasets: list[str] | None = None,
) -> OutcomeMatrix:
    """Load record trees under `root` (the extracted bench-release dir) into an OutcomeMatrix."""
    model_names = models if models is not None else FLAGSHIP_MODELS
    dataset_dirs = sorted(
        d for d in root.iterdir() if d.is_dir() and (datasets is None or d.name in datasets)
    )
    if not dataset_dirs:
        raise ValueError(f"no dataset dirs under {root}")

    pool = [
        PoolEntry(
            name=name,
            kind=ProviderKind.OPENAI,
            model=name,
            tier="frontier",
            # Costs live on each outcome (measured per record); see the RouterBench adapter.
            input_per_mtok=0.0,
            output_per_mtok=0.0,
        )
        for name in model_names
    ]

    outcomes: list[ScenarioOutcome] = []
    # Task-level leakage control: several dataset dirs share query texts (the arenahard
    # subsets categorize the same prompts), so a scenario split without text dedupe leaks
    # fit tasks into test - live-caught inflating a learned router by ~+12pt (2026-07-24).
    # First occurrence wins; drops are logged, never silent.
    seen_tasks: set[str] = set()
    duplicate_scenarios = 0
    for dataset_dir in dataset_dirs:
        per_model: dict[str, dict[int, dict]] = {}
        for name in model_names:
            model_dir = _model_dir(dataset_dir, name)
            if model_dir is not None:
                records = _load_records(model_dir)
                if records:
                    per_model[name] = records
        if len(per_model) < len(model_names):
            missing = sorted(set(model_names) - set(per_model))
            logger.info("%s: skipped (missing models: %s)", dataset_dir.name, missing[:4])
            continue
        shared = set.intersection(*(set(r) for r in per_model.values()))
        dropped = max(len(r) for r in per_model.values()) - len(shared)
        if dropped:
            logger.info(
                "%s: %d records not covered by all models, dropped", dataset_dir.name, dropped
            )
        for index in sorted(shared):
            query = str(per_model[model_names[0]][index]["origin_query"])
            if query in seen_tasks:
                duplicate_scenarios += 1
                continue
            seen_tasks.add(query)
            sid = f"{dataset_dir.name}:{index}"
            for name in model_names:
                rec = per_model[name][index]
                score = float(rec["score"])
                outcomes.append(
                    ScenarioOutcome(
                        scenario_id=sid,
                        task=query,
                        model=name,
                        reward=score,
                        success=score >= 0.5,
                        steps=1,
                        stop_reason="llmrouterbench",
                        cost_usd=float(rec["cost"]),
                    )
                )
    if not outcomes:
        raise ValueError(f"no datasets under {root} covered all of: {model_names}")
    if duplicate_scenarios:
        logger.info("dropped %d duplicate-text scenarios (leakage control)", duplicate_scenarios)
    return OutcomeMatrix(pool=pool, outcomes=outcomes)
