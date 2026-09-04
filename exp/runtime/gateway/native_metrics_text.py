"""Prometheus text rendering of the content-free gateway metrics snapshot.

The native data plane serves ``GET /metrics.json`` from the composed snapshot
built by ``NativeControlPlane.metrics_snapshot``. This module renders that
same snapshot in the Prometheus text exposition format (version 0.0.4) so the
``GET /metrics`` route can feed a standard scraper without a sidecar. The
rendering is a pure function of the snapshot: every family is a fixed name
with bounded enum labels, so the exposition stays content-free on the same
terms as the JSON body (no alias, key, model, prompt, or org identifier).
"""

from __future__ import annotations

from typing import cast

from exp.common.core.artifacts import JsonObject

METRICS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
"""Content type of the Prometheus text exposition served at ``/metrics``."""

_PREFIX = "exp_gateway_"

_DATA_PLANE_COUNTERS: tuple[tuple[str, str], ...] = (
    ("served_requests", "Requests admitted and served natively."),
    ("open_retries", "Same-deployment retries at the upstream open phase."),
    ("settlement_retries", "Settlement deliveries retried after a failed write."),
    ("settlement_give_ups", "Settlements whose bounded retries were all exhausted."),
)

_DATA_PLANE_GAUGES: tuple[tuple[str, str], ...] = (
    ("active_requests", "Natively served requests currently in the data plane."),
)

_HISTOGRAMS: tuple[tuple[str, str], ...] = (
    ("time_to_first_byte_ms", "Milliseconds from admission to the first upstream byte."),
    ("request_duration_ms", "Total milliseconds spent serving one request."),
    ("permit_wait_ms", "Milliseconds a request waited for an admission permit."),
    ("bridge_call_ms", "Milliseconds one control-plane bridge call took."),
)

_CONTROL_PLANE_COUNTERS: tuple[tuple[str, str], ...] = (
    (
        "sweep_retained_settlements_replayed",
        "Settlements replayed by the sweep.",
    ),
    (
        "sweep_abandoned_attempts_cancelled",
        "Abandoned attempts the sweep cancelled.",
    ),
    (
        "admission_dead_rungs_skipped",
        "Certified rungs skipped as dead at admission.",
    ),
    (
        "admission_lead_rungs_skipped",
        "Fallbacks served for a dead lead rung.",
    ),
    (
        "admission_parameter_coercions",
        "Disclosed request coercions at admission.",
    ),
    (
        "rung_admission_sheds",
        "Dispatches shed by a rung's bound or fair share.",
    ),
    (
        "rung_saturated_overflows",
        "Dispatches forced past a saturated rung bound.",
    ),
    (
        "reconciled_expired_requests",
        "Crashed requests reconciled at startup.",
    ),
    (
        "reconciled_unknown_attempts",
        "Crashed attempts reconciled at startup.",
    ),
)


def _value(number: object) -> str:
    """Render one sample value in exposition syntax.

    Args:
        number: Integer, float, or boolean sample value from the snapshot.

    Returns:
        The value as Prometheus text: booleans as ``1``/``0``, integers
        plainly, floats through ``repr`` so no precision is invented.
    """
    if isinstance(number, bool):
        return "1" if number else "0"
    if isinstance(number, int):
        return str(number)
    if isinstance(number, float):
        return repr(number)
    raise TypeError(f"metric value must be numeric, got {type(number).__name__}")


def _family(name: str, kind: str, help_text: str, samples: list[str]) -> list[str]:
    """Compose one metric family's HELP, TYPE, and sample lines.

    Args:
        name: Full metric family name including the prefix.
        kind: Exposition metric type: ``counter``, ``gauge``, or ``histogram``.
        help_text: One content-free sentence describing the family.
        samples: Fully rendered sample lines for the family.

    Returns:
        The family's exposition lines in order.
    """
    return [f"# HELP {name} {help_text}", f"# TYPE {name} {kind}", *samples]


def _labeled_counter(section: JsonObject, key: str, label: str, help_text: str) -> list[str]:
    """Render one enum-labeled counter family from a snapshot sub-object.

    Args:
        section: Snapshot section holding the enum-to-count object at ``key``.
        key: Snapshot key of the enum-to-count object.
        label: Label name carrying the enum value.
        help_text: One content-free sentence describing the family.

    Returns:
        The family's exposition lines, one sample per enum value.
    """
    name = f"{_PREFIX}{key}_total"
    counts = cast("JsonObject", section[key])
    samples = [f'{name}{{{label}="{entry}"}} {_value(count)}' for entry, count in counts.items()]
    return _family(name, "counter", help_text, samples)


def _histogram(section: JsonObject, key: str, help_text: str) -> list[str]:
    """Render one fixed-bucket histogram family in cumulative bucket form.

    The snapshot stores per-bucket counts with millisecond upper bounds and a
    ``null`` bound for the overflow bucket; the exposition requires cumulative
    ``_bucket`` samples ending in ``le="+Inf"`` equal to ``_count``. The
    registry samples each atomic independently under concurrent recording, so
    ``_count`` is emitted from the bucket total itself rather than the
    separately sampled snapshot count, keeping every scrape internally
    consistent.

    Args:
        section: Snapshot section holding the histogram object at ``key``.
        key: Snapshot key of the histogram object.
        help_text: One content-free sentence describing the family.

    Returns:
        The family's ``_bucket``, ``_sum``, and ``_count`` exposition lines.
    """
    name = f"{_PREFIX}{key}"
    histogram = cast("JsonObject", section[key])
    samples: list[str] = []
    cumulative = 0
    for bucket in cast("list[JsonObject]", histogram["buckets"]):
        cumulative += cast("int", bucket["count"])
        upper = bucket["le_ms"]
        bound = "+Inf" if upper is None else str(upper)
        samples.append(f'{name}_bucket{{le="{bound}"}} {cumulative}')
    samples.append(f"{name}_sum {_value(histogram['sum_ms'])}")
    samples.append(f"{name}_count {cumulative}")
    return _family(name, "histogram", help_text, samples)


def render_metrics_text(snapshot: JsonObject) -> str:
    """Render the composed metrics snapshot as Prometheus exposition text.

    Args:
        snapshot: The ``metrics_snapshot`` body with a ``control_plane``
            section and a ``data_plane`` section that is ``None`` when no
            native registry provider is bound (hosted without the callback).

    Returns:
        The exposition text: data-plane families first when present, then the
        control-plane families, ending with one trailing newline.
    """
    lines: list[str] = []
    data_plane = cast("JsonObject | None", snapshot.get("data_plane"))
    if data_plane is not None:
        lines += _labeled_counter(
            data_plane,
            "requests",
            "outcome",
            "Terminal outcomes of natively served requests.",
        )
        lines += _labeled_counter(
            data_plane,
            "escalated_requests",
            "kind",
            "Escalated admissions by bounded kind.",
        )
        for key, help_text in _DATA_PLANE_COUNTERS:
            lines += _family(
                f"{_PREFIX}{key}_total",
                "counter",
                help_text,
                [f"{_PREFIX}{key}_total {_value(data_plane[key])}"],
            )
        for key, help_text in _DATA_PLANE_GAUGES:
            lines += _family(
                f"{_PREFIX}{key}",
                "gauge",
                help_text,
                [f"{_PREFIX}{key} {_value(data_plane[key])}"],
            )
        for key, help_text in _HISTOGRAMS:
            lines += _histogram(data_plane, key, help_text)
    control_plane = cast("JsonObject", snapshot["control_plane"])
    for key, help_text in _CONTROL_PLANE_COUNTERS:
        lines += _family(
            f"{_PREFIX}{key}_total",
            "counter",
            help_text,
            [f"{_PREFIX}{key}_total {_value(control_plane[key])}"],
        )
    lines += _family(
        f"{_PREFIX}inflight_attempts",
        "gauge",
        "In-flight attempt reservations on the control plane.",
        [f"{_PREFIX}inflight_attempts {_value(control_plane['inflight_attempts'])}"],
    )
    lines += _family(
        f"{_PREFIX}accounting_healthy",
        "gauge",
        "Control-plane accounting health (1 healthy, 0 failed).",
        [f"{_PREFIX}accounting_healthy {_value(control_plane['accounting_healthy'])}"],
    )
    return "\n".join(lines) + "\n"
