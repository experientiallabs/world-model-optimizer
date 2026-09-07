"""Resolve one declared trace source name to its canonical loader.

Ingestion owns the set of supported sources, so a caller names a source and passes a local path
instead of importing one loader per vendor. The mapping is an explicit table in this module: there
is no import-time registration, no plugin discovery, and no format detection. A name the table does
not declare fails closed.

    result = load_trace_source("langfuse", Path("export.jsonl"))

Every loader returns the same ``TraceNormalizationResult``, and every failure that is specific to a
source (unreadable bytes, malformed payloads) is raised as ``TraceSourceError`` so a caller does
not have to know which vendor errors exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from exp.simulation.ingest.braintrust import BRAINTRUST_SOURCE
from exp.simulation.ingest.chat_json import CHAT_JSON_SOURCE
from exp.simulation.ingest.datadog import DATADOG_SOURCE
from exp.simulation.ingest.langfuse import LANGFUSE_SOURCE
from exp.simulation.ingest.langsmith import LANGSMITH_SOURCE
from exp.simulation.ingest.mastra import MASTRA_SOURCE
from exp.simulation.ingest.otel_genai import load_otel_genai_file
from exp.simulation.ingest.otlp import (
    OtlpTraceFormatError,
    TraceNormalizationResult,
    load_otlp_file,
)
from exp.simulation.ingest.phoenix import PHOENIX_SOURCE
from exp.simulation.ingest.posthog import PostHogPullError, load_posthog_file
from exp.simulation.ingest.vendor_records import VendorTraceFormatError


class TraceSourceError(ValueError):
    """Raised when a source name is unsupported or its declared corpus cannot be normalized."""


class _TraceFileLoader(Protocol):
    """Canonical file loader that can receive a caller-owned durable source label."""

    def __call__(
        self,
        path: Path,
        *,
        source_id: str | None = None,
    ) -> TraceNormalizationResult:
        """Normalize one local file while preserving the supplied source identity."""
        ...


_LOADERS: dict[str, _TraceFileLoader] = {
    "braintrust": BRAINTRUST_SOURCE.load,
    "chat-json": CHAT_JSON_SOURCE.load,
    "datadog": DATADOG_SOURCE.load,
    "langfuse": LANGFUSE_SOURCE.load,
    "langsmith": LANGSMITH_SOURCE.load,
    "mastra": MASTRA_SOURCE.load,
    "otel-genai": load_otel_genai_file,
    "otlp": load_otlp_file,
    "phoenix": PHOENIX_SOURCE.load,
    "posthog": load_posthog_file,
}
CANONICAL_TRACE_SOURCES: tuple[str, ...] = tuple(sorted(_LOADERS))


def load_trace_source(
    source: str,
    path: Path,
    *,
    source_id: str | None = None,
) -> TraceNormalizationResult:
    """Normalize one local corpus through the loader of its declared source.

    Args:
        source: Declared source name, matched case-insensitively after trimming.
        path: Local trace export.
        source_id: Optional durable label that replaces the worker-local path in provenance.

    Returns:
        Canonical traces and every retained validation exclusion.

    Raises:
        TraceSourceError: The source is unsupported or its corpus cannot be normalized.
    """
    loader = _LOADERS.get(source.strip().casefold())
    if loader is None:
        choices = ", ".join(CANONICAL_TRACE_SOURCES)
        raise TraceSourceError(f"unsupported trace source {source!r}; choose one of: {choices}")
    try:
        return loader(path, source_id=source_id)
    except (
        OtlpTraceFormatError,
        PostHogPullError,
        VendorTraceFormatError,
        ValueError,
    ) as exc:
        raise TraceSourceError(f"{source.strip().casefold()} normalization failed: {exc}") from None
