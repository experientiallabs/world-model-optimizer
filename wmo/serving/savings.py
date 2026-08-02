"""What an endpoint has saved so far, computed from its own request log.

The customer-facing counterpart of the cost/quality dial: the dial says what the endpoint is
TRYING to do, this says what it has actually done since it started serving. Cost is the honest
part (logged dollars against a priced counterfactual), latency is an estimate calibrated on the
endpoint's own traffic, and quality is a fitted expectation carried over from the offline
measurement, clearly labeled as such because live quality needs a feedback signal nobody is
sending yet.

Everything is recomputed from the persisted JSONL rows rather than accumulated in memory, so a
restart does not reset a customer's savings and a number can always be traced back to the rows
that produced it.

Every estimate carries its basis as a sentence in `estimate_basis`. Those strings render verbatim
in the customer UI, so they are written for a customer: no knob names, no internals, and no
number without a stated basis.
"""

from __future__ import annotations

import statistics
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from wmo.optimize.knn import COST_QUALITY_ANCHORS, COST_QUALITY_BALANCED, cost_quality_knobs
from wmo.providers.base import TokenUsage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wmo.optimize.policy import RoutingPolicy
    from wmo.serving.chat import RequestLogRecord

SavingsWindow = Literal["all_time", "7d"]

WINDOW_DAYS = 7

# The sentences the response ships. Written as customer copy on purpose (see module docstring).
BASIS_COUNTERFACTUAL = (
    "Savings compare what you were billed against what the same requests would have cost on "
    "{fallback} alone, priced on the same number of tokens each request actually used, including "
    "crediting {fallback} with the cached reads the model that served earned. Both assumptions "
    "understate the saving rather than inflate it."
)
BASIS_NO_TRAFFIC = "This endpoint has not served any requests yet, so there is nothing to compare."
BASIS_LATENCY_SELF = (
    "Time saved is an estimate: it compares each request that used a different model against "
    "the median response time of this endpoint's own {fallback} requests, so it becomes more "
    "accurate as the endpoint serves more traffic."
)
BASIS_LATENCY_NO_BASELINE = (
    "Time saved is not shown yet: this endpoint has not served enough requests on {fallback} to "
    "establish a response time to compare against."
)
BASIS_QUALITY_ANCHOR = (
    "The quality figure is a fitted expectation from offline evaluation of this setting, not a "
    "live measurement of your traffic."
)
BASIS_QUALITY_INTERPOLATED = (
    "The quality figure is interpolated between the two nearest evaluated settings, so treat it "
    "as a direction rather than a precise value."
)
BASIS_QUALITY_AS_FITTED = (
    "This endpoint is serving the settings it was optimized with, which are our balanced "
    "setting, so the quality figure is that setting's fitted expectation from offline "
    "evaluation rather than a live measurement of your traffic."
)
BASIS_QUALITY_UNKNOWN = (
    "No quality figure is shown: this endpoint was tuned by hand to settings we have not "
    "evaluated, so there is no fitted expectation to quote for it."
)
BASIS_QUALITY_NO_DIAL = (
    "No quality figure is shown: this endpoint sends every request to one model, so there is no "
    "cost and quality setting to compare against."
)
BASIS_BILLING = "Your invoices remain the record of what you were charged."


class EndpointSavings(BaseModel):
    """One endpoint's savings over one window, as the platform card renders it.

    `requests_served` is a count of successfully served requests; a card with 0 there is the
    empty state, and every other field is zero rather than null so a client never has to
    special-case a missing key. `cost_saved_usd` and `cost_saved_pct` come from logged dollars
    against the priced counterfactual; the two accounting fields behind them are included so the
    subtraction is auditable. `time_saved_s_estimate` and `expected_quality_delta_pt` are
    estimates, named so, and every estimate's basis is a sentence in `estimate_basis`.
    """

    requests_served: int = Field(ge=0)
    cost_saved_usd: float
    cost_saved_pct: float
    time_saved_s_estimate: float
    expected_quality_delta_pt: float
    estimate_basis: list[str]
    window: SavingsWindow
    # The two sums the cost saving is the difference of: logged spend (provider bill plus any
    # compressor bill, per the track's effective-cost rule), and the counterfactual.
    actual_cost_usd: float = Field(ge=0.0)
    baseline_cost_estimate_usd: float = Field(ge=0.0)


def _in_window(record: RequestLogRecord, *, window: SavingsWindow, now: datetime) -> bool:
    if window == "all_time":
        return True
    try:
        stamped = datetime.fromisoformat(record.ts)
    except ValueError:
        # An unparseable timestamp cannot be placed in a bounded window. It still counts toward
        # all-time, where no placement is needed.
        return False
    if stamped.tzinfo is None:
        stamped = stamped.replace(tzinfo=UTC)
    return stamped >= now - timedelta(days=WINDOW_DAYS)


def _expected_quality(policy: RoutingPolicy) -> tuple[float, str]:
    """The fitted quality expectation for the endpoint's dial position, and its basis.

    Exactly on an anchor: that anchor's measured delta. Between two anchors: a linear
    interpolation, labeled as one. Dial never set: the balanced anchor, because a policy fitted
    with the shipped defaults IS the balanced setting, which is checked against ALL FOUR knobs
    the dial controls rather than assumed. The coverage knob matters most here: `floor_q` is the
    only thing separating the balanced setting from the quality-max one, so a fit that set it
    differently is a different operating point no matter how the other three read. A policy off
    the dial, or one whose `floor_q` was never recorded, gets no figure at all: quoting the
    balanced number for an endpoint someone tuned elsewhere would be a claim about an evaluation
    that never ran.
    """
    dial = policy.cost_quality
    balanced = next(
        anchor
        for anchor in COST_QUALITY_ANCHORS
        if abs(anchor.cost_quality - COST_QUALITY_BALANCED) < 1e-9
    )
    if policy.kind != "knn":
        return 0.0, BASIS_QUALITY_NO_DIAL
    if dial is None:
        default_knobs = cost_quality_knobs(COST_QUALITY_BALANCED)
        as_fitted_is_balanced = (
            policy.floor_q is not None
            and abs(policy.floor_q - default_knobs.floor_q) < 1e-9
            and policy.knn_z == default_knobs.knn_z
            and policy.pick_lam == default_knobs.pick_lam
            and policy.guard_mode == default_knobs.guard_mode
        )
        if as_fitted_is_balanced:
            return balanced.quality_delta_points, BASIS_QUALITY_AS_FITTED
        return 0.0, BASIS_QUALITY_UNKNOWN
    anchors = sorted(COST_QUALITY_ANCHORS, key=lambda anchor: anchor.cost_quality)
    for anchor in anchors:
        if abs(anchor.cost_quality - dial) < 1e-9:
            return anchor.quality_delta_points, BASIS_QUALITY_ANCHOR
    below = [anchor for anchor in anchors if anchor.cost_quality < dial]
    above = [anchor for anchor in anchors if anchor.cost_quality > dial]
    if not below or not above:
        # Unreachable while the anchors span the full dial (a test pins that they do); if that
        # ever changes, say nothing rather than extrapolate off the end of the measurement.
        return 0.0, BASIS_QUALITY_UNKNOWN
    low, high = below[-1], above[0]
    span = high.cost_quality - low.cost_quality
    weight = (dial - low.cost_quality) / span
    delta = (1.0 - weight) * low.quality_delta_points + weight * high.quality_delta_points
    return delta, BASIS_QUALITY_INTERPOLATED


def compute_savings(
    records: Sequence[RequestLogRecord],
    policy: RoutingPolicy,
    *,
    window: SavingsWindow = "all_time",
    now: datetime | None = None,
) -> EndpointSavings:
    """Total up what this endpoint saved over `window`, from its logged rows.

    The counterfactual is `policy.guard_model` (the fallback the endpoint would have served
    every request on without a policy), priced by its own pool entry on each request's ACTUAL
    token counts. That is an assumption, not a measurement: a different model would have emitted
    a different number of output tokens, and its prompt cache would have been its own. It is the
    assumption the response states, and it is the conservative direction for a router that
    routes toward cheaper models, since those models tend to be the wordier ones.

    Latency has no counterfactual price list, so its baseline is measured instead: the median
    latency of this endpoint's OWN fallback-served requests. That self-calibrates as traffic
    accrues, and until there are fallback requests to take a median of, no figure is reported.
    Differences are summed signed, so a routed model that ran slower subtracts. Requests that
    failed are excluded from every total: nobody was served.

    The compressor's own bill is part of what the endpoint spent. The compression track's
    accounting rule is explicit about it ("every savings number is cache-adjusted effective cost
    per completed task, compressor cost and latency included"), so `compressor_cost_usd` is
    added to the actual side and never to the counterfactual: the fallback-only baseline runs no
    compressor. Crediting the token reduction while omitting what the reduction cost would
    inflate every compressed endpoint's savings by the compressor's entire bill. Compressor
    latency needs no separate handling: `latency_ms` already spans the compression stage.
    """
    entries = {entry.name: entry for entry in policy.pool}
    fallback = policy.guard_model or policy.default_model
    served = [
        record
        for record in records
        if record.status == "ok" and _in_window(record, window=window, now=now or datetime.now(UTC))
    ]
    basis: list[str] = []
    if not served:
        return EndpointSavings(
            requests_served=0,
            cost_saved_usd=0.0,
            cost_saved_pct=0.0,
            time_saved_s_estimate=0.0,
            expected_quality_delta_pt=0.0,
            estimate_basis=[BASIS_NO_TRAFFIC, BASIS_BILLING],
            window=window,
            actual_cost_usd=0.0,
            baseline_cost_estimate_usd=0.0,
        )

    actual = sum(record.cost_usd + record.compressor_cost_usd for record in served)
    baseline_entry = entries.get(fallback)
    baseline = actual
    if baseline_entry is not None:
        baseline = sum(
            baseline_entry.call_cost_usd(
                TokenUsage(
                    input_tokens=record.input_tokens,
                    output_tokens=record.output_tokens,
                    cached_input_tokens=record.cached_tokens,
                    cache_write_input_tokens=record.cache_write_tokens,
                )
            )
            for record in served
        )
    basis.append(BASIS_COUNTERFACTUAL.format(fallback=fallback))

    on_fallback = [record.latency_ms for record in served if record.model == fallback]
    routed_away = [record for record in served if record.model != fallback]
    time_saved = 0.0
    if on_fallback and routed_away:
        fallback_p50 = statistics.median(on_fallback)
        time_saved = sum(fallback_p50 - record.latency_ms for record in routed_away) / 1000.0
        basis.append(BASIS_LATENCY_SELF.format(fallback=fallback))
    elif routed_away:
        basis.append(BASIS_LATENCY_NO_BASELINE.format(fallback=fallback))

    quality, quality_basis = _expected_quality(policy)
    basis.append(quality_basis)
    basis.append(BASIS_BILLING)
    return EndpointSavings(
        requests_served=len(served),
        cost_saved_usd=baseline - actual,
        cost_saved_pct=((baseline - actual) / baseline * 100.0) if baseline > 0.0 else 0.0,
        time_saved_s_estimate=time_saved,
        expected_quality_delta_pt=quality,
        estimate_basis=basis,
        window=window,
        actual_cost_usd=actual,
        baseline_cost_estimate_usd=baseline,
    )
