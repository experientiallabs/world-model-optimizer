"""Closed-loop outcome matrix: per (scenario x candidate model x episode) eval records.

This is the routing optimizer's training data and the improvement report's evidence base, the
RouterBench-style precomputed matrix: run the pool over the scenario set once, then compare any
number of policy variants offline on identical data. Produced by `wmo.env.closed_loop`
(kept import-free of `wmo.env`/`wmo.engine` here so optimizers can consume it without cycles).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel, model_validator

from wmo.core.files import write_text_atomic
from wmo.optimize.compression import CompressionConfig
from wmo.providers.base import TokenUsage
from wmo.providers.pool import PoolEntry

# Provenance carries a digest of the matrix, not just its path: a corpus is routinely rebuilt in
# place under the same filename, and a fit is identified by the data it saw. 16 hex characters
# is 64 bits, far past collision risk for the handful of matrices one artifact directory sees.
MATRIX_DIGEST_CHARS = 16

# Router fitting and reporting share one paid outcome matrix, but they must not share scenarios.
# Hash ordering makes the partition stable across machines and independent of matrix row order.
ROUTER_FIT_FRACTION = 0.7
ROUTER_SPLIT_VERSION = "scenario-hash-70-30-v1"


class RouterScenarioSplit(BaseModel):
    """Disjoint scenario ids used to fit a router and report its generalization."""

    fit_ids: tuple[str, ...]
    report_ids: tuple[str, ...]


def split_router_scenarios(scenario_ids: list[str]) -> RouterScenarioSplit:
    """Deterministically reserve 30% of scenarios for router evaluation.

    At least two scenarios are required because a router trained and evaluated on one scenario
    cannot produce a held-out claim. Returned ids retain the matrix's original order; hashing is
    used only to assign membership.
    """
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("router split requires unique scenario ids")
    if len(scenario_ids) < 2:
        raise ValueError(
            "router fitting needs at least 2 scenarios so one can remain held out for reporting"
        )
    ranked = sorted(
        scenario_ids,
        key=lambda scenario_id: (
            hashlib.sha256(scenario_id.encode("utf-8")).digest(),
            scenario_id,
        ),
    )
    fit_count = round(len(ranked) * ROUTER_FIT_FRACTION)
    fit_count = min(len(ranked) - 1, max(1, fit_count))
    fit_set = set(ranked[:fit_count])
    return RouterScenarioSplit(
        fit_ids=tuple(sid for sid in scenario_ids if sid in fit_set),
        report_ids=tuple(sid for sid in scenario_ids if sid not in fit_set),
    )


class ScenarioOutcome(BaseModel):
    """One episode of one candidate model on one scenario.

    `reward` is None only for unscored episodes (`error` says why); consumers fitting policies
    or reporting numbers must skip unscored rows, never default them to 0 (an infrastructure
    failure is not a judge verdict).
    """

    scenario_id: str
    task: str
    model: str  # pool entry name (the stable handle policy artifacts key on)
    benchmark: str = ""
    episode: int = 0
    attempt_number: int = 1
    reward: float | None = None
    success: bool = False
    critique: str = ""
    steps: int = 0
    tool_calls: int = 0
    stop_reason: str = ""
    usage: TokenUsage = TokenUsage()
    cost_usd: float = 0.0  # candidate-side cost, priced by ITS pool entry
    call_seconds: list[float] = []  # wall seconds per policy call (env time excluded)
    # Per-call token counts retain request-level pricing boundaries. Empty on legacy matrices.
    call_input_tokens: list[int] = []
    call_output_tokens: list[int] = []
    call_cached_input_tokens: list[int] = []
    call_cache_write_input_tokens: list[int] = []
    # Provider counters are preferred. Experiment runners may preserve a gradeable official
    # result with a clearly labeled trace-derived estimate when exact counters are unavailable.
    usage_accounting: str = "exact"
    usage_estimate_method: str = ""
    wall_seconds: float = 0.0
    completion_status: str = ""
    failure_class: str = ""
    artifact_dir: str = ""
    # Raw completion texts per call: the future distillation feed (stored, never used by v1
    # fitting). Reasoning models that emit thought before the JSON action keep it here.
    replies: list[str] = []
    error: str | None = None
    # Whether this row is a RE-measurement of a cell an earlier attempt already ran (a transport
    # fault, a throttled judge). Additive with a default so pre-retry matrices load unchanged.
    # Retries are not free evidence: a cell measured only on its second try survived a filter its
    # neighbours never faced, and a reader comparing two matrices deserves to see how much of one
    # of them is re-runs.
    remeasured: bool = False
    # D-COMPRESS fields, additive with defaults so pre-compression matrices load unchanged;
    # 0/"" = the episode ran uncompressed. Token counts are the compressor's deterministic
    # proxy totals summed over the episode's calls (wmo.optimize.compression); billable truth
    # stays in `usage`/`cost_usd`. Latency and cost are the compressor's OWN, which sit inside
    # effective cost per the compression track's accounting rules.
    tokens_in_raw: int = 0
    tokens_in_compressed: int = 0
    compressor_id: str = ""
    compressor_version: str = ""
    aggressiveness: float = 0.0
    compressor_latency_s: float = 0.0
    compressor_cost_usd: float = 0.0

    @property
    def scored(self) -> bool:
        return self.reward is not None


class OutcomeMatrix(BaseModel):
    """The full pool x scenario outcome grid, plus the pool it was measured on.

    Carrying the pool snapshot makes the matrix self-describing: a policy fitted from it can
    record exactly which candidates (and at what prices) its assignments were chosen over.
    """

    pool: list[PoolEntry]
    outcomes: list[ScenarioOutcome]

    @model_validator(mode="after")
    def _outcomes_name_pool_models(self) -> OutcomeMatrix:
        """Every outcome must name a pool entry, or the matrix is not self-describing.

        Consumers index the pool by outcome model (`fit_rank_policy`'s `pool_order`, the report's
        per-candidate table). A row naming a model the pool never heard of used to surface as a
        bare `KeyError` deep inside a fitter; caught here it names the offender instead.
        """
        names = {entry.name for entry in self.pool}
        ghosts = sorted({o.model for o in self.outcomes if o.model not in names})
        if ghosts:
            raise ValueError(
                f"outcomes name models missing from the pool: {ghosts[:5]}; "
                f"pool models are {sorted(names)}"
            )
        return self

    def measured_compression(self) -> CompressionConfig | None:
        """The compression config every scored row was measured under (None = uncompressed).

        A matrix is ONE arm of the grid: its rewards were produced by episodes that all ran
        under the same conditions, which is what makes the rows comparable and what a policy
        fitted from them is entitled to claim. Rows that disagree about compression are two arms
        in one file, so this raises rather than picking a winner.

        Reads the scored rows only: an unscored row never produced a reward, so whatever it ran
        under cannot bias a fit.
        """
        configs = {
            (o.compressor_id, o.compressor_version, o.aggressiveness)
            for o in self.outcomes
            if o.scored
        }
        if len(configs) > 1:
            readable = sorted(f"{cid or 'uncompressed'}/{ver}/{agg:g}" for cid, ver, agg in configs)
            raise ValueError(
                "this matrix mixes compression configs across its scored rows "
                f"({', '.join(readable)}), so its rows are not comparable and no single policy "
                "can be fitted from them. Capture one matrix per arm."
            )
        if not configs:
            return None
        compressor_id, compressor_version, aggressiveness = configs.pop()
        if not compressor_id:
            return None
        return CompressionConfig(
            compressor_id=compressor_id,
            compressor_version=compressor_version,
            aggressiveness=aggressiveness,
        )

    def model_names(self) -> list[str]:
        return [entry.name for entry in self.pool]

    def scenario_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for outcome in self.outcomes:
            seen.setdefault(outcome.scenario_id, None)
        return list(seen)

    def for_scenario(self, scenario_id: str) -> list[ScenarioOutcome]:
        return [o for o in self.outcomes if o.scenario_id == scenario_id]

    def mean_reward(self, model: str) -> float:
        """Mean reward of `model` over its scored episodes."""
        if model not in self.model_names():
            raise KeyError(f"no pool model named '{model}'; available: {self.model_names()}")
        rewards = [o.reward for o in self.outcomes if o.model == model and o.reward is not None]
        if not rewards:
            raise ValueError(f"pool model '{model}' has no scored episodes in this matrix")
        return sum(rewards) / len(rewards)

    def save(self, path: Path) -> None:
        """Write the matrix atomically: it is the measured output of a run that cost real money."""
        write_text_atomic(path, self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> OutcomeMatrix:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def load_matrix_with_digest(matrix_file: Path) -> tuple[OutcomeMatrix, str]:
    """The matrix and its `<path> sha256=<digest>` provenance, from ONE read of the file.

    The digest is what makes a policy's `fitted_from` an identity rather than a label. `tune`
    compares it against the as-fitted snapshot beside a policy, and two fits of the same path with
    different contents (or the same contents at two paths) have to come out different for that
    check to protect anything.

    Both come out of the same bytes on purpose. Digesting a SECOND read would let a corpus
    rebuilt in place between the two describe the fit as having seen bytes it never saw: the
    policy would be fitted from the old matrix and stamped with the new one's digest, so the
    next fit of that new matrix would match its provenance and `tune` would accept the
    superseded snapshot, the exact failure the digest exists to catch, reintroduced by
    reading twice.
    """
    payload = matrix_file.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    return (
        OutcomeMatrix.model_validate_json(payload),
        f"{matrix_file} sha256={digest[:MATRIX_DIGEST_CHARS]}",
    )
