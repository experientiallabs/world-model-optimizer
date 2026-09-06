"""Renderer tests: exact exposition text, grammar validity, absent data plane."""

from __future__ import annotations

import re
from typing import cast

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.native_metrics_text import METRICS_CONTENT_TYPE, render_metrics_text

_BUCKET_UPPER_MS = (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000)


def _histogram(per_bucket: list[int], sum_ms: float, count: int) -> JsonObject:
    """Build one snapshot histogram in the native registry's JSON shape.

    Args:
        per_bucket: Non-cumulative counts for the fifteen bounded buckets.
        sum_ms: Total observed milliseconds.
        count: Total observation count; the overflow bucket absorbs the
            remainder beyond the bounded buckets.

    Returns:
        The histogram object with ``count``, ``sum_ms``, and ``buckets``.
    """
    buckets: list[JsonObject] = [
        {"le_ms": upper, "count": observed}
        for upper, observed in zip(_BUCKET_UPPER_MS, per_bucket, strict=True)
    ]
    buckets.append({"le_ms": None, "count": count - sum(per_bucket)})
    return {"count": count, "sum_ms": sum_ms, "buckets": buckets}


def _control_plane() -> JsonObject:
    """Build the control-plane section shared by both fixture snapshots."""
    return {
        "sweep_retained_settlements_replayed": 1,
        "sweep_abandoned_attempts_cancelled": 0,
        "admission_dead_rungs_skipped": 0,
        "admission_lead_rungs_skipped": 0,
        "admission_parameter_coercions": 0,
        "rung_admission_sheds": 0,
        "rung_saturated_overflows": 0,
        "inflight_attempts": 2,
        "reconciled_expired_requests": 0,
        "reconciled_unknown_attempts": 0,
        "accounting_healthy": True,
    }


def _snapshot() -> JsonObject:
    """Build the fixed full snapshot fixture with a populated data plane."""
    empty = _histogram([0] * 15, 0.0, 0)
    return {
        "data_plane": {
            "requests": {"completed": 3, "incomplete": 1, "failed": 2, "cancelled": 1},
            "served_requests": 7,
            "escalated_requests": {
                "project_alias": 1,
                "deployment_pool": 0,
                "provider_dialect": 2,
                "host_policy": 0,
                "other": 0,
            },
            "open_retries": 2,
            "settlement_retries": 1,
            "settlement_give_ups": 0,
            "active_requests": 1,
            "time_to_first_byte_ms": _histogram([2, 1] + [0] * 13, 120007.5, 4),
            "request_duration_ms": empty,
            "permit_wait_ms": empty,
            "bridge_call_ms": empty,
        },
        "control_plane": _control_plane(),
    }


def _empty_histogram_lines(name: str, help_text: str) -> str:
    """Render the expected exposition block for one empty histogram.

    Args:
        name: Full family name including the prefix.
        help_text: The family's HELP sentence.

    Returns:
        The family's expected lines, newline-joined with a trailing newline.
    """
    bounds = (*_BUCKET_UPPER_MS, "+Inf")
    buckets = "".join(f'{name}_bucket{{le="{upper}"}} 0\n' for upper in bounds)
    header = f"# HELP {name} {help_text}\n# TYPE {name} histogram\n"
    return f"{header}{buckets}{name}_sum 0.0\n{name}_count 0\n"


_EXPECTED_CONTROL_PLANE = """\
# HELP exp_gateway_sweep_retained_settlements_replayed_total Settlements replayed by the sweep.
# TYPE exp_gateway_sweep_retained_settlements_replayed_total counter
exp_gateway_sweep_retained_settlements_replayed_total 1
# HELP exp_gateway_sweep_abandoned_attempts_cancelled_total Abandoned attempts the sweep cancelled.
# TYPE exp_gateway_sweep_abandoned_attempts_cancelled_total counter
exp_gateway_sweep_abandoned_attempts_cancelled_total 0
# HELP exp_gateway_admission_dead_rungs_skipped_total Certified rungs skipped as dead at admission.
# TYPE exp_gateway_admission_dead_rungs_skipped_total counter
exp_gateway_admission_dead_rungs_skipped_total 0
# HELP exp_gateway_admission_lead_rungs_skipped_total Fallbacks served for a dead lead rung.
# TYPE exp_gateway_admission_lead_rungs_skipped_total counter
exp_gateway_admission_lead_rungs_skipped_total 0
# HELP exp_gateway_admission_parameter_coercions_total Disclosed request coercions at admission.
# TYPE exp_gateway_admission_parameter_coercions_total counter
exp_gateway_admission_parameter_coercions_total 0
# HELP exp_gateway_rung_admission_sheds_total Dispatches shed by a rung's bound or fair share.
# TYPE exp_gateway_rung_admission_sheds_total counter
exp_gateway_rung_admission_sheds_total 0
# HELP exp_gateway_rung_saturated_overflows_total Dispatches forced past a saturated rung bound.
# TYPE exp_gateway_rung_saturated_overflows_total counter
exp_gateway_rung_saturated_overflows_total 0
# HELP exp_gateway_reconciled_expired_requests_total Crashed requests reconciled at startup.
# TYPE exp_gateway_reconciled_expired_requests_total counter
exp_gateway_reconciled_expired_requests_total 0
# HELP exp_gateway_reconciled_unknown_attempts_total Crashed attempts reconciled at startup.
# TYPE exp_gateway_reconciled_unknown_attempts_total counter
exp_gateway_reconciled_unknown_attempts_total 0
# HELP exp_gateway_inflight_attempts In-flight attempt reservations on the control plane.
# TYPE exp_gateway_inflight_attempts gauge
exp_gateway_inflight_attempts 2
# HELP exp_gateway_accounting_healthy Control-plane accounting health (1 healthy, 0 failed).
# TYPE exp_gateway_accounting_healthy gauge
exp_gateway_accounting_healthy 1
"""

_EXPECTED_FULL = (
    """\
# HELP exp_gateway_requests_total Terminal outcomes of natively served requests.
# TYPE exp_gateway_requests_total counter
exp_gateway_requests_total{outcome="completed"} 3
exp_gateway_requests_total{outcome="incomplete"} 1
exp_gateway_requests_total{outcome="failed"} 2
exp_gateway_requests_total{outcome="cancelled"} 1
# HELP exp_gateway_escalated_requests_total Escalated admissions by bounded kind.
# TYPE exp_gateway_escalated_requests_total counter
exp_gateway_escalated_requests_total{kind="project_alias"} 1
exp_gateway_escalated_requests_total{kind="deployment_pool"} 0
exp_gateway_escalated_requests_total{kind="provider_dialect"} 2
exp_gateway_escalated_requests_total{kind="host_policy"} 0
exp_gateway_escalated_requests_total{kind="other"} 0
# HELP exp_gateway_served_requests_total Requests admitted and served natively.
# TYPE exp_gateway_served_requests_total counter
exp_gateway_served_requests_total 7
# HELP exp_gateway_open_retries_total Same-deployment retries at the upstream open phase.
# TYPE exp_gateway_open_retries_total counter
exp_gateway_open_retries_total 2
# HELP exp_gateway_settlement_retries_total Settlement deliveries retried after a failed write.
# TYPE exp_gateway_settlement_retries_total counter
exp_gateway_settlement_retries_total 1
# HELP exp_gateway_settlement_give_ups_total Settlements whose bounded retries were all exhausted.
# TYPE exp_gateway_settlement_give_ups_total counter
exp_gateway_settlement_give_ups_total 0
# HELP exp_gateway_active_requests Natively served requests currently in the data plane.
# TYPE exp_gateway_active_requests gauge
exp_gateway_active_requests 1
# HELP exp_gateway_time_to_first_byte_ms Milliseconds from admission to the first upstream byte.
# TYPE exp_gateway_time_to_first_byte_ms histogram
exp_gateway_time_to_first_byte_ms_bucket{le="1"} 2
exp_gateway_time_to_first_byte_ms_bucket{le="2"} 3
exp_gateway_time_to_first_byte_ms_bucket{le="5"} 3
exp_gateway_time_to_first_byte_ms_bucket{le="10"} 3
exp_gateway_time_to_first_byte_ms_bucket{le="25"} 3
exp_gateway_time_to_first_byte_ms_bucket{le="50"} 3
exp_gateway_time_to_first_byte_ms_bucket{le="100"} 3
exp_gateway_time_to_first_byte_ms_bucket{le="250"} 3
exp_gateway_time_to_first_byte_ms_bucket{le="500"} 3
exp_gateway_time_to_first_byte_ms_bucket{le="1000"} 3
exp_gateway_time_to_first_byte_ms_bucket{le="2500"} 3
exp_gateway_time_to_first_byte_ms_bucket{le="5000"} 3
exp_gateway_time_to_first_byte_ms_bucket{le="10000"} 3
exp_gateway_time_to_first_byte_ms_bucket{le="30000"} 3
exp_gateway_time_to_first_byte_ms_bucket{le="60000"} 3
exp_gateway_time_to_first_byte_ms_bucket{le="+Inf"} 4
exp_gateway_time_to_first_byte_ms_sum 120007.5
exp_gateway_time_to_first_byte_ms_count 4
"""
    + _empty_histogram_lines(
        "exp_gateway_request_duration_ms", "Total milliseconds spent serving one request."
    )
    + _empty_histogram_lines(
        "exp_gateway_permit_wait_ms", "Milliseconds a request waited for an admission permit."
    )
    + _empty_histogram_lines(
        "exp_gateway_bridge_call_ms", "Milliseconds one control-plane bridge call took."
    )
    + _EXPECTED_CONTROL_PLANE
)

_HELP_OR_TYPE = re.compile(
    r"^# (HELP [a-zA-Z_:][a-zA-Z0-9_:]* \S.*"
    r"|TYPE [a-zA-Z_:][a-zA-Z0-9_:]* (counter|gauge|histogram))$"
)
_SAMPLE = re.compile(
    r"^[a-zA-Z_:][a-zA-Z0-9_:]*"
    r'(\{[a-zA-Z_][a-zA-Z0-9_]*="[^"\\\n]*"(,[a-zA-Z_][a-zA-Z0-9_]*="[^"\\\n]*")*\})?'
    r" (\+Inf|-?[0-9]+(\.[0-9]+)?)$"
)


def test_renderer_matches_the_exact_expected_exposition() -> None:
    """The fixed snapshot fixture renders to the exact expected text."""
    assert render_metrics_text(_snapshot()) == _EXPECTED_FULL


def test_every_line_satisfies_the_exposition_grammar() -> None:
    """Every rendered line is a valid comment or sample per the text format."""
    rendered = render_metrics_text(_snapshot())
    assert rendered.endswith("\n")
    for line in rendered.rstrip("\n").split("\n"):
        assert _HELP_OR_TYPE.match(line) or _SAMPLE.match(line), line


def test_counters_end_in_total_and_histograms_close_at_inf() -> None:
    """Naming rules hold: counters end _total; +Inf buckets equal _count."""
    rendered = render_metrics_text(_snapshot())
    lines = rendered.rstrip("\n").split("\n")
    for line in lines:
        if line.startswith("# TYPE ") and line.endswith(" counter"):
            assert line.split(" ")[2].endswith("_total"), line
    inf_counts = {
        line.split("_bucket")[0]: line.rsplit(" ", 1)[1]
        for line in lines
        if '_bucket{le="+Inf"}' in line
    }
    totals = {
        line.rsplit(" ", 1)[0].removesuffix("_count"): line.rsplit(" ", 1)[1]
        for line in lines
        if not line.startswith("#") and line.rsplit(" ", 1)[0].endswith("_ms_count")
    }
    assert inf_counts == totals
    assert len(inf_counts) == 4


def test_a_torn_snapshot_still_renders_an_internally_consistent_histogram() -> None:
    """A count sampled behind the buckets never breaks +Inf equals _count.

    The registry samples each atomic independently, so a concurrent recording
    can leave the snapshot's total one behind the bucket counters. The
    exposition derives _count from the bucket total itself.
    """
    snapshot = _snapshot()
    data_plane = cast("JsonObject", snapshot["data_plane"])
    torn = cast("JsonObject", data_plane["time_to_first_byte_ms"])
    torn["count"] = 3
    rendered = render_metrics_text(snapshot)
    assert 'exp_gateway_time_to_first_byte_ms_bucket{le="+Inf"} 4' in rendered
    assert "exp_gateway_time_to_first_byte_ms_count 4" in rendered


def test_absent_data_plane_renders_only_control_plane_metrics() -> None:
    """A hosted snapshot without the callback renders control-plane families only."""
    rendered = render_metrics_text({"data_plane": None, "control_plane": _control_plane()})
    assert rendered == _EXPECTED_CONTROL_PLANE
    assert "exp_gateway_requests_total" not in rendered
    assert "_bucket" not in rendered


def test_non_numeric_values_are_rejected() -> None:
    """A non-numeric snapshot value raises instead of rendering invalid text."""
    control = _control_plane()
    control["inflight_attempts"] = "two"
    with pytest.raises(TypeError):
        render_metrics_text({"data_plane": None, "control_plane": control})


def test_the_exposition_content_type_is_the_prometheus_text_version() -> None:
    """The published content type pins text format version 0.0.4."""
    assert METRICS_CONTENT_TYPE == "text/plain; version=0.0.4; charset=utf-8"
