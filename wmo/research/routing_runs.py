"""Uniform run records for routing ablations: one evaluator, every variant, full transparency.

Every ablation run (static, best-single, rank, rank+knob, IRT, future variants) evaluates
through `evaluate_choices` - a `choose(scenario_id) -> model` callable - so metrics are
computed identically regardless of the router's internals. Each run persists as one JSONL
record (`append_run`) carrying the EXPLAIN BLOCK alongside the headline numbers: model mix,
per-model latency, and the blended token breakdown by model. That is what makes results like
"cost went down AND p50 went down" verifiable instead of fishy: cheap models are usually also
the fast ones, and the mix-weighted per-model latency table shows it directly.

The dashboard and run reports read these records; serving's request log (D-METERING) is the
live twin of the same fields.
"""

from __future__ import annotations

from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, JsonValue

from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome

if TYPE_CHECKING:
    from collections.abc import Callable


class ChoiceEval(BaseModel):
    """Metrics + explain block for one routing over a scenario set."""

    accuracy: float
    cost_per_call: float  # for multi-call policies this is cost per SCENARIO (all calls summed)
    latency_p50_s: float | None = None  # None when the matrix carries no timings
    latency_p95_s: float | None = None
    scenarios: int
    unscored: int
    model_mix: dict[str, float]
    tokens_by_model: dict[str, dict[str, int]]  # {model: {input, output}}
    per_model_latency_p50_s: dict[str, float] = Field(default_factory=dict)
    per_model_cost_per_call: dict[str, float] = Field(default_factory=dict)
    calls_per_scenario: float = 1.0  # >1 for cascades / best-of-n


class RunRecord(BaseModel):
    """One ablation run, as persisted to runs.jsonl (the dashboard's data source)."""

    run_id: str
    ts: str
    matrix: str  # which outcome matrix (routerbench, llmrouterbench, ours, wm corpus name)
    variant: str  # static | best-single | rank | irt | ...
    params: dict[str, JsonValue] = Field(default_factory=dict)
    split_seed: int = 0
    fit_scenarios: int = 0
    test_scenarios: int = 0
    result: ChoiceEval
    baselines: dict[str, ChoiceEval] = Field(default_factory=dict)
    notes: str = ""


def evaluate_choices(
    matrix: OutcomeMatrix, ids: list[str], choose: Callable[[str], str]
) -> ChoiceEval:
    """Score `choose` over `ids` with the full explain block (see module docstring)."""
    by_cell: dict[tuple[str, str], list] = {}
    for outcome in matrix.outcomes:
        if outcome.reward is not None:
            by_cell.setdefault((outcome.scenario_id, outcome.model), []).append(outcome)

    rewards: list[float] = []
    costs: list[float] = []
    seconds: list[float] = []
    mix: dict[str, int] = {}
    tokens: dict[str, dict[str, int]] = {}
    per_model_seconds: dict[str, list[float]] = {}
    per_model_costs: dict[str, list[float]] = {}
    unscored = 0
    for sid in ids:
        model = choose(sid)
        mix[model] = mix.get(model, 0) + 1
        cells = by_cell.get((sid, model))
        if not cells:
            unscored += 1
            continue
        rewards.append(sum(o.reward for o in cells if o.reward is not None) / len(cells))
        cell_cost = sum(o.cost_usd for o in cells) / len(cells)
        costs.append(cell_cost)
        per_model_costs.setdefault(model, []).append(cell_cost)
        bucket = tokens.setdefault(model, {"input": 0, "output": 0})
        for outcome in cells:
            bucket["input"] += outcome.usage.input_tokens
            bucket["output"] += outcome.usage.output_tokens
            for value in outcome.call_seconds:
                seconds.append(value)
                per_model_seconds.setdefault(model, []).append(value)
    if not rewards:
        raise ValueError("no scored outcomes for any routed choice")

    def _p95(values: list[float]) -> float:
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]

    return ChoiceEval(
        accuracy=sum(rewards) / len(rewards),
        cost_per_call=sum(costs) / len(costs),
        latency_p50_s=median(seconds) if seconds else None,
        latency_p95_s=_p95(seconds) if seconds else None,
        scenarios=len(ids),
        unscored=unscored,
        model_mix={m: count / len(ids) for m, count in sorted(mix.items())},
        tokens_by_model=tokens,
        per_model_latency_p50_s={
            m: median(values) for m, values in sorted(per_model_seconds.items())
        },
        per_model_cost_per_call={
            m: sum(values) / len(values) for m, values in sorted(per_model_costs.items())
        },
    )


class Finish(BaseModel):
    """A multi-call policy's stop decision: `pick` indexes the transcript call whose result
    is the scenario's answer (default -1: the most recent call)."""

    pick: int = -1


def evaluate_call_sequences(
    matrix: OutcomeMatrix,
    ids: list[str],
    policy: Callable[[str, list[ScenarioOutcome]], str | Finish],
    *,
    max_calls: int = 8,
) -> ChoiceEval:
    """Score a MULTI-call policy (cascade, best-of-n, ensemble) over `ids`.

    The policy is called repeatedly per scenario with the transcript of outcomes so far: return
    a model name to issue the next call, or `Finish(pick=i)` to stop and answer with call `i`.
    The k-th call to a model consumes that model's k-th stored episode (which is exactly what
    makes best-of-2 simulable from a 2-episode matrix); a call to a model with no scored episode
    left marks the scenario unscored. Accuracy uses only the PICKED call's reward; cost sums
    ALL calls; latency pools per-LLM-call timings (comparable to `evaluate_choices`, NOT
    end-to-end sequential latency).

    Information boundary: transcript entries expose `reward` and `critique` (the judge's
    verdict). A policy that reads them at decision time is an ORACLE simulation - an upper
    bound, never a deployable router - and its runs must be labeled `oracle-*`. Deployable
    policies may only read `replies`, `steps`, `stop_reason`, `usage`, `cost_usd`,
    `call_seconds`.

    COMPARABILITY TRAP, do not skip. A row from here is comparable to an `evaluate_choices` row
    ONLY when the policy's value is order-independent. `evaluate_choices` averages a cell's
    episodes, which is the expected value of one call; this function consumes the k-th episode, so
    an order-DEPENDENT policy (anything that answers with a fixed call index, a 1-call sequence
    being the simplest case) scores one specific draw instead. Scoring a 1-call fallback here made
    it read 10pt worse than the identical best-single policy on terminal-tasks, purely from
    episode-0 luck. If a policy's answer depends on call order, either score its 1x arm through
    `evaluate_choices` or average over episode permutations; otherwise every paired delta against
    best-single carries that offset. Selection rules that rank the transcript by a feature (the
    best-of-n selectors in `wmo.research.posthoc_bounds`) are order-independent and safe.
    """
    by_cell: dict[tuple[str, str], list[ScenarioOutcome]] = {}
    for outcome in matrix.outcomes:
        if outcome.reward is not None:
            by_cell.setdefault((outcome.scenario_id, outcome.model), []).append(outcome)
    for cell in by_cell.values():
        cell.sort(key=lambda o: o.episode)

    rewards: list[float] = []
    costs: list[float] = []
    seconds: list[float] = []
    mix: dict[str, int] = {}
    tokens: dict[str, dict[str, int]] = {}
    per_model_seconds: dict[str, list[float]] = {}
    per_model_costs: dict[str, list[float]] = {}
    unscored = 0
    total_calls = 0
    for sid in ids:
        transcript: list[ScenarioOutcome] = []
        used: dict[str, int] = {}
        finish: Finish | None = None
        while True:
            decision = policy(sid, list(transcript))
            if isinstance(decision, Finish):
                finish = decision
                break
            if len(transcript) >= max_calls:
                raise ValueError(
                    f"policy exceeded max_calls={max_calls} on scenario {sid!r} "
                    "without returning Finish"
                )
            episode_index = used.get(decision, 0)
            used[decision] = episode_index + 1
            cell = by_cell.get((sid, decision), [])
            if episode_index >= len(cell):
                break  # no scored episode left for this model on this scenario
            transcript.append(cell[episode_index])
        if finish is None or not transcript:
            unscored += 1
            continue
        picked = transcript[finish.pick]  # IndexError = policy bug, let it surface
        rewards.append(picked.reward if picked.reward is not None else 0.0)
        costs.append(sum(o.cost_usd for o in transcript))
        total_calls += len(transcript)
        for outcome in transcript:
            mix[outcome.model] = mix.get(outcome.model, 0) + 1
            per_model_costs.setdefault(outcome.model, []).append(outcome.cost_usd)
            bucket = tokens.setdefault(outcome.model, {"input": 0, "output": 0})
            bucket["input"] += outcome.usage.input_tokens
            bucket["output"] += outcome.usage.output_tokens
            for value in outcome.call_seconds:
                seconds.append(value)
                per_model_seconds.setdefault(outcome.model, []).append(value)
    if not rewards:
        raise ValueError("no scored outcomes for any routed call sequence")

    def _p95(values: list[float]) -> float:
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]

    return ChoiceEval(
        accuracy=sum(rewards) / len(rewards),
        cost_per_call=sum(costs) / len(costs),
        latency_p50_s=median(seconds) if seconds else None,
        latency_p95_s=_p95(seconds) if seconds else None,
        scenarios=len(ids),
        unscored=unscored,
        model_mix={m: count / total_calls for m, count in sorted(mix.items())},
        tokens_by_model=tokens,
        per_model_latency_p50_s={
            m: median(values) for m, values in sorted(per_model_seconds.items())
        },
        per_model_cost_per_call={
            m: sum(values) / len(values) for m, values in sorted(per_model_costs.items())
        },
        calls_per_scenario=total_calls / len(rewards),
    )


def append_run(record: RunRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(record.model_dump_json() + "\n")


def run_report(record: RunRecord) -> str:
    """Human-readable run report: headline, deltas vs baselines, and the explain block."""
    result = record.result
    lines = [
        f"# Run {record.run_id}: {record.variant} on {record.matrix}",
        f"{record.ts} | params {record.params} | split seed {record.split_seed} | "
        f"{result.scenarios} test scenarios ({result.unscored} unscored)",
        "",
        f"accuracy {result.accuracy:.4f} | cost/call ${result.cost_per_call:.5f}"
        + (
            f" | latency p50 {result.latency_p50_s:.2f}s p95 {result.latency_p95_s:.2f}s"
            if result.latency_p50_s is not None
            else " | latency: not measured on this matrix"
        ),
    ]
    for name, baseline in record.baselines.items():
        cost_delta = (result.cost_per_call / baseline.cost_per_call - 1) * 100
        lines.append(
            f"vs {name}: accuracy {result.accuracy - baseline.accuracy:+.4f}, "
            f"cost {cost_delta:+.1f}%"
            + (
                f", p50 {result.latency_p50_s - baseline.latency_p50_s:+.2f}s"
                if result.latency_p50_s is not None and baseline.latency_p50_s is not None
                else ""
            )
        )
    lines += ["", "## Why (explain block)"]
    if result.calls_per_scenario > 1.0:
        lines.append(
            f"Multi-call policy: {result.calls_per_scenario:.2f} calls per scenario on "
            "average; cost/call above is per SCENARIO (all calls summed), and the mix below "
            "is over calls, not scenarios."
        )
    lines.append("Model mix, with each model's own latency and cost:")
    for model, share in result.model_mix.items():
        p50 = result.per_model_latency_p50_s.get(model)
        cost = result.per_model_cost_per_call.get(model)
        lines.append(
            f"- {model}: {share:.0%} of calls"
            + (f", p50 {p50:.2f}s" if p50 is not None else "")
            + (f", ${cost:.5f}/call" if cost is not None else "")
        )
    lines.append("")
    lines.append("Blended tokens by model (input/output):")
    for model, bucket in result.tokens_by_model.items():
        lines.append(f"- {model}: {bucket['input']:,} in / {bucket['output']:,} out")
    if result.latency_p50_s is not None:
        lines.append("")
        lines.append(
            "Note: cost and latency can drop TOGETHER when the mix shifts toward cheaper "
            "models, because cheaper models are usually also faster (see per-model p50 above)."
        )
    lines.append("")
    if record.notes:
        lines.append(record.notes)
    return "\n".join(lines)
