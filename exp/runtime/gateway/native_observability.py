"""Discovery, usage, metrics, and lifecycle callbacks for the native bridge.

:class:`NativeObservabilityMixin` carries the content-free read-side
callbacks of :class:`~exp.runtime.gateway.native_bridge.NativeControlPlane`:
granted-model discovery, the usage report, the composed metrics snapshot,
readiness, and per-worker resource release. The mixin owns no state; it reads
the control plane's bound components and accounting registry, so the bridge
module keeps only the request-serving authority path.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import cast

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.discovery import (
    public_model_list,
    public_model_object,
    require_granted_authority,
)
from exp.runtime.gateway.ledger import SQLiteAttemptLedger
from exp.runtime.gateway.native_accounting import (
    NativeAttemptAccounting,
)
from exp.runtime.gateway.native_accounting import (
    authority_error as _authority_error,
)
from exp.runtime.gateway.native_components import NativeGatewayComponents
from exp.runtime.gateway.native_metrics_text import render_metrics_text
from exp.runtime.gateway.sqlite.migrations import close_idle_connections
from exp.runtime.gateway.usage import GatewayUsageReport, read_usage_report, usage_html


class NativeObservabilityMixin:
    """Content-free read-side callbacks shared by the native control plane."""

    _components: NativeGatewayComponents
    _accounting: NativeAttemptAccounting
    _data_plane_metrics: Callable[[], str] | None
    _usage_reporter: Callable[[], JsonObject] | None
    _readiness_probe: Callable[[], bool] | None

    def models(self, argument: str) -> str:
        """Return the granted model list body for one authenticated key."""
        data = json.loads(argument)
        try:
            authorities = self._components.store.granted_alias_authorities(raw_key=data["raw_key"])
            body = public_model_list(authorities)
        except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
            raise _authority_error(exc) from exc
        return json.dumps(body, separators=(",", ":"))

    def model_detail(self, argument: str) -> str:
        """Return one granted model object or the shared no-oracle 404."""
        data = json.loads(argument)
        try:
            authorities = self._components.store.granted_alias_authorities(raw_key=data["raw_key"])
            authority = require_granted_authority(authorities, data["model_id"])
            body = public_model_object(authority)
        except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
            raise _authority_error(exc) from exc
        return json.dumps(body, separators=(",", ":"))

    def usage_json(self, argument: str) -> str:
        """Return the content-free usage report body.

        Args:
            argument: JSON object with an optional ``raw_key``. A presented
                key scopes the report to its owning identity; an absent key
                returns the organization-wide report.

        Returns:
            The schema-versioned usage report as one JSON object.

        Raises:
            NativeBridgeError: The presented key is invalid, expired, or revoked.
        """
        if self._usage_reporter is not None:
            return json.dumps(self._usage_reporter(), separators=(",", ":"))
        report = self._usage_report(argument)
        return json.dumps(report.model_dump(mode="json"), separators=(",", ":"))

    def usage_page(self, argument: str) -> str:
        """Return the content-free usage page rendering.

        Args:
            argument: JSON object with an optional ``raw_key``, scoped exactly
                like :meth:`usage_json`.

        Returns:
            JSON object with one ``html`` field holding the rendered page.

        Raises:
            NativeBridgeError: The presented key is invalid, expired, or revoked.
        """
        report = self._usage_report(argument)
        return json.dumps({"html": usage_html(report)}, separators=(",", ":"))

    def _usage_report(self, argument: str) -> GatewayUsageReport:
        """Read the usage report for one optionally key-scoped callback.

        Args:
            argument: JSON object with an optional ``raw_key``.

        Returns:
            The organization-wide report, or the report scoped to the
            presented key's identity.

        Raises:
            NativeBridgeError: The presented key is invalid, expired, or revoked.
        """
        data = json.loads(argument)
        raw_key = data.get("raw_key")
        identity_id: str | None = None
        if raw_key is not None:
            try:
                _, identity_id = self._components.store.authenticated_identity(raw_key=str(raw_key))
            except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
                raise _authority_error(exc) from exc
        # Only the local composition (SQLite ledger) serves native usage;
        # hosted compositions disable it and own their own usage surface.
        return read_usage_report(
            cast("SQLiteAttemptLedger", self._components.ledger),
            organization_id=self._components.organization_id,
            identity_id=identity_id,
        )

    def metrics_snapshot(self) -> JsonObject:
        """Compose the one content-free observability snapshot.

        ``data_plane`` carries the native engine's registry when a provider
        is bound, otherwise ``None``; ``control_plane`` carries this bridge's
        own sweep recoveries, in-flight registry size, reconciliation counts,
        and accounting health. The native ``/metrics.json`` route serves
        exactly this body; ``/metrics`` serves its Prometheus text rendering.
        """
        data_plane: JsonObject | None = None
        if self._data_plane_metrics is not None:
            data_plane = json.loads(self._data_plane_metrics())
        retained_replayed, abandoned_cancelled, inflight = self._accounting.counters()
        lead_rungs_skipped, dead_rungs_skipped = self._accounting.admission_rung_skips()
        rung_sheds, rung_overflows = self._accounting.rung_admission_counters()
        control_plane: JsonObject = {
            "sweep_retained_settlements_replayed": retained_replayed,
            "sweep_abandoned_attempts_cancelled": abandoned_cancelled,
            "admission_dead_rungs_skipped": dead_rungs_skipped,
            "admission_lead_rungs_skipped": lead_rungs_skipped,
            "admission_parameter_coercions": self._accounting.admission_parameter_coercions(),
            "rung_admission_sheds": rung_sheds,
            "rung_saturated_overflows": rung_overflows,
            "inflight_attempts": inflight,
            "reconciled_expired_requests": self._components.reconciled_expired_requests,
            "reconciled_unknown_attempts": self._components.reconciled_unknown_attempts,
            "accounting_healthy": self._accounting.accounting_healthy,
        }
        return {"data_plane": data_plane, "control_plane": control_plane}

    def metrics_json(self, argument: str) -> str:
        """Return the content-free metrics snapshot body for the data plane."""
        del argument
        return json.dumps(self.metrics_snapshot(), separators=(",", ":"))

    def metrics_text(self, argument: str) -> str:
        """Return ``{"text": ...}`` holding the snapshot's Prometheus exposition."""
        del argument
        text = render_metrics_text(self.metrics_snapshot())
        return json.dumps({"text": text}, separators=(",", ":"))

    def readiness(self, argument: str) -> str:
        """Return whether bridge settlement and composition accounting stay healthy.

        Readiness fails closed once the bridge's own settlement registry has
        latched a durable-write loss, once the composition's accounting
        surface reports unhealthy, or when an injected hosted lifecycle probe
        answers false or raises.
        """
        del argument
        if not self._accounting.accounting_healthy:
            return "false"
        try:
            if not self._components.accounting_healthy:
                return "false"
        except Exception:  # noqa: BLE001 - readiness fails closed at the boundary.
            return "false"
        if self._readiness_probe is not None:
            try:
                return "true" if self._readiness_probe() else "false"
            except Exception:  # noqa: BLE001 - readiness fails closed at the boundary.
                return "false"
        return "true"

    def close_thread_resources(self, argument: str) -> str:
        """Release the calling thread's cached resources before it exits.

        The data plane pins control-plane callbacks to a fixed pool of worker
        threads and calls this once per worker as the pool shuts down, so the
        SQLite connections cached per thread by ``persistent_connection``
        close with the worker instead of outliving it.

        Args:
            argument: Empty JSON object, matching the callback shape.

        Returns:
            JSON object reporting the number of connections closed.
        """
        del argument
        return json.dumps({"closed_connections": close_idle_connections()}, separators=(",", ":"))
